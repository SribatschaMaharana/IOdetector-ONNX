import numpy as np
import torchvision.transforms.v2 as trn
from PIL import Image
import onnxruntime as ort


class IOClassifierProcessing:
    def __init__(self):
        self.classes, self.labels_IO = self.load_labels()
        self.valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    def load_labels(self):
        # prepare labels
        classes = []
        file_name_category = "categories_places365.txt"
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(" ")[0][3:])
        classes = tuple(classes)

        file_name_IO = "IO_places365.txt"
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        return classes, labels_IO

    def returnTF(self):
        # image transformer
        tf = trn.Compose(
            [
                trn.Resize((224, 224)),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tf

    def preprocess_single(self, image_path):
        # preprocess a single image
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        tf = self.returnTF()
        input_img = tf(img).unsqueeze(0).numpy()  # to numpy for onnx runtime param type
        return input_img

    def postprocess_single(self, output, image_path):
        # after forward pass? exact interaction of softmax - check
        probs = output
        idx = np.argsort(probs)[::-1]  # sort indices by descending probability

        #  calc i/o score
        io_image = np.average(self.labels_IO[idx[:10]], weights=probs[idx[:10]])

        result = {
            "Image": image_path,
            "Environment Type": "Indoor" if io_image < 0.5 else "Outdoor",
        }
        scene = []
        for i in range(5):
            if probs[idx[i]] > 0.01:
                scene.append(
                    {
                        "Description": self.classes[idx[i]],
                        "Confidence": str(round(float(probs[idx[i]]), 3)),
                    }
                )
        result["Scene Category"] = scene

        return result


class IOClassifierModel:
    def __init__(self, model_path):
        self.processor = IOClassifierProcessing()
        self.session = ort.InferenceSession(
            model_path,
            providers=[
                "CPUExecutionProvider"
            ],  # "CUDAExecutionProvider" add this param if needed for GPU exec
        )

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, image_path):
        input = self.processor.preprocess_single(image_path)
        output = self.session.run(None, {"input": input})
        probs = self._softmax(output[0][0])  # redundant if softmax is already applied
        result = self.processor.postprocess_single(probs, image_path)
        return result
