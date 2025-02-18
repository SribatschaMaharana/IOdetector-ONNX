import warnings
import json
from typing import TypedDict
import os

from flask_ml.flask_ml_server import MLServer, load_file_as_string

from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    FileInput,
    FileResponse,
    FileType,
    InputSchema,
    InputType,
    NewFileInputType,
    ResponseBody,
    TaskSchema,
)

from onnx_helper import IOClassifierModel

warnings.filterwarnings("ignore")

class ImageInputs(TypedDict):
    image_input: BatchFileInput
    output_path: FileInput

class ImageParameters(TypedDict):
    pass

server = MLServer(__name__)
model = IOClassifierModel("iodetector.onnx")

server.add_app_metadata(
    name="Indoor/Outdoor Classifier - ONNX",
    author="Sribatscha Maharana",
    version="0.2.0",
    info=load_file_as_string("README.md"),
)

def image_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="image_input", 
                label="Upload images", 
                input_type=InputType.BATCHFILE
            ),
            InputSchema(
                key="output_path",
                label="Output JSON Path",
                input_type=NewFileInputType(
                    default_name="output.json",
                    default_extension=".json",
                    allowed_extensions=[".json"],
                ),
            ),
        ],
        parameters=[],
    )

def save_to_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

@server.route(
    "/process_images", task_schema_func=image_task_schema, short_title="Result"
)
def process_images(inputs: ImageInputs, parameters: ImageParameters) -> ResponseBody:
    # get file paths
    output_path = inputs["output_path"].path
    input_dir = inputs["image_input"].files
    if os.path.exists(output_path):
        os.remove(output_path)

    #predict images here 
    results = []
    for file in input_dir:
        result = model.predict(file.path)
        results.append(result)
    
    save_to_json(output_path, results)
    print(f"Results written to: {output_path}")
    
    return ResponseBody(
        root=FileResponse(
            file_type=FileType.JSON,
            path=output_path,
            title="Indoor/Outdoor Classification Report",
        )
    )

if __name__ == "__main__":
    server.run()