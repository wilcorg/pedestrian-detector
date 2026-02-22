"""
Roman Szlachtun 272330

Kopia poleceń z YOLO-export.ipynb

Konwertowanie modelu YOLO z PyTorch do OpenVINO.
"""

from ultralytics import YOLO
from pathlib import Path

modelPath = Path('/content/drive/MyDrive/Colab Notebooks/960p_color_100it_29_09__20_17/detect/train/weights/best.pt')
modelToExport = YOLO(str(modelPath))

modelToExport.export(
    format="openvino",
    imgsz=960,
    data='/content/drive/MyDrive/datasets/otc/data.yaml',
    int8=True,
    nms=False,
    rect=True,
)

"""
Ultralytics 8.3.228 🚀 Python-3.12.12 torch-2.8.0+cu126 CPU (Intel Xeon CPU @ 2.00GHz)
YOLO11n summary (fused): 100 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs

PyTorch: starting from '/content/drive/MyDrive/Colab Notebooks/960p_color_100it_29_09__20_17/detect/train/weights/best.pt' with input shape (1, 3, 960, 960) BCHW and output shape(s) (1, 5, 18900) (5.2 MB)

OpenVINO: starting export with openvino 2025.3.0-19807-44526285f24-releases/2025/3...
OpenVINO: collecting INT8 calibration images from 'data=/content/drive/MyDrive/datasets/otc/data.yaml'
Fast image access ✅ (ping: 6.1±10.5 ms, read: 68.7±67.6 MB/s, size: 1348.4 KB)
Scanning /content/drive/MyDrive/datasets/otc/labels/val.cache... 450 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 450/450 451.8Kit/s 0.0s
INFO:nncf:15 ignored nodes were found by patterns in the NNCFGraph
INFO:nncf:1 ignored nodes were found by types in the NNCFGraph
INFO:nncf:Not adding activation input quantizer for operation: 168 __module.model.23.dfl/aten::view/Reshape
INFO:nncf:Not adding activation input quantizer for operation: 169 __module.model.23/aten::sigmoid/Sigmoid
INFO:nncf:Not adding activation input quantizer for operation: 180 __module.model.23.dfl/aten::transpose/Transpose
INFO:nncf:Not adding activation input quantizer for operation: 191 __module.model.23.dfl/aten::softmax/Softmax
INFO:nncf:Not adding activation input quantizer for operation: 200 __module.model.23.dfl.conv/aten::_convolution/Convolution
INFO:nncf:Not adding activation input quantizer for operation: 208 __module.model.23.dfl/aten::view/Reshape_1
INFO:nncf:Not adding activation input quantizer for operation: 226 __module.model.23/aten::sub/Subtract
INFO:nncf:Not adding activation input quantizer for operation: 227 __module.model.23/aten::add/Add
INFO:nncf:Not adding activation input quantizer for operation: 236 __module.model.23/aten::add/Add_1
246 __module.model.23/aten::div/Divide

INFO:nncf:Not adding activation input quantizer for operation: 237 __module.model.23/aten::sub/Subtract_1
INFO:nncf:Not adding activation input quantizer for operation: 256 __module.model.23/aten::mul/Multiply

OpenVINO: export success ✅ 244.2s, saved as '/content/drive/MyDrive/Colab Notebooks/960p_color_100it_29_09__20_17/detect/train/weights/best_int8_openvino_model/' (3.3 MB)

Export complete (244.9s)
Results saved to /content/drive/MyDrive/Colab Notebooks/960p_color_100it_29_09__20_17/detect/train/weights
Predict:         yolo predict task=detect model=/content/drive/MyDrive/Colab Notebooks/960p_color_100it_29_09__20_17/detect/train/weights/best_int8_openvino_model imgsz=960 int8 
Validate:        yolo val task=detect model=/content/drive/MyDrive/Colab Notebooks/960p_color_100it_29_09__20_17/detect/train/weights/best_int8_openvino_model imgsz=960 data=/content/drive/MyDrive/datasets/otc/data.yaml int8 
Visualize:       https://netron.app
"""