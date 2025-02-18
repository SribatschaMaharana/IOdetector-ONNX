import warnings

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileInput,
    FileResponse,
    FileType,
    InputSchema,
    InputType,
    NewFileInputType,
    ResponseBody,
    TaskSchema,
)

# Import model helper
from onnx_helper import IOClassifierModel

warnings.filterwarnings("ignore")