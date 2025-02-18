import argparse
from pathlib import Path
import json
from pprint import pprint
from onnx_helper import IOClassifierModel

# parse
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to an image or directory containing images")
parser.add_argument("--output", type=str, required=True, help="Path to an output JSON file, or directory for output to reside in")
args = parser.parse_args()

# get inputs
input_path = Path(args.input)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

model = IOClassifierModel("iodetector.onnx")

results = []

# some processing for prediction
if input_path.is_file():
    result = model.predict(str(input_path))
    results.append(result)
elif input_path.is_dir():
    valid_extensions = model.processor.valid_extensions
    
    for img_path in input_path.glob("**/*"):
        if img_path.suffix.lower() in valid_extensions:
            try:
                result = model.predict(str(img_path))
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
else:
    print(f"Error: input path '{input_path}' not found or not accessible")
    exit(1)

pprint(results)

if output_path.is_dir():
    output_path = output_path / "output.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nResults saved to {output_path}")
