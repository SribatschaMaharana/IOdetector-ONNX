# IOdetector-ONNX

This repository contains an ONNX version of the Indoor/Outdoor scene classifier model, designed to work with the UMass RescueBox frontend. The model classifies images as either indoor or outdoor scenes and provides the top scene categories.

## Steps to export the ONNX model

Exporting the ONNX model involved the following key steps:
1. Clone and set up the UMass-Rescue/GeoLocator repo found [here](https://github.com/UMass-Rescue/GeoLocator). Follow instructions on README.md to get a dry run of the model working.
2. Configure [Rescue Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop/releases) to work with the GeoLocator repo. To test that the application works, send an example request where the input are files from "input" folder from this (IOdetector-ONNX) repo. Once you get the GeoLocator model working, proceed to the next step. (note, an ```import torch``` command is missing in the flaskml-server.py script, required)
3. Set a breakpoint at Line 195, right before the `logit = model.forward(input_img)` call in the `run_iodetector` function in `IndoorOutdoorClassifier/iodetector.py` by adding the following line: `import pdb; pdb.set_trace()`.
4. Send a request to the GeoLocator backend again using the same inputs from the RescueBox Desktop application. The breakpoint will be triggered in the backend.
5. Run the following python code to export the ONNX model.
```
torch.onnx.export(
    model,              
    input_img,        
    "iodetector.onnx",   
    export_params=True, 
    opset_version=16,    
    do_constant_folding=True, 
    input_names=["input"],  
    output_names=["output"],   
    dynamic_axes={ "input": {0: "batch_size"},"output": {0: "batch_size"}}
)
```

The resulting ONNX model will be saved as "iodetector.onnx" in the directory where the `flaskml-server.py` exists.
