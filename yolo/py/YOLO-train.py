"""
Roman Szlachtun 272330

Kopia poleceń z YOLO-train.ipynb

Trenowanie modelu YOLO.
"""


#!nvidia-smi
"""
Mon Nov 10 19:24:31 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:00:04.0 Off |                    0 |
| N/A   35C    P0             51W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
"""

#!pip install ultralytics

from IPython import display
display.clear_output()

#!yolo checks

"""
Ultralytics 8.3.227 🚀 Python-3.12.12 torch-2.8.0+cu126 CUDA:0 (NVIDIA A100-SXM4-40GB, 40507MiB)
Setup complete ✅ (12 CPUs, 83.5 GB RAM, 45.9/235.7 GB disk)

OS                     Linux-6.6.105+-x86_64-with-glibc2.35
Environment            Colab
Python                 3.12.12
Install                pip
Path                   /usr/local/lib/python3.12/dist-packages/ultralytics
RAM                    83.47 GB
Disk                   45.9/235.7 GB
CPU                    Intel Xeon CPU @ 2.20GHz
CPU count              12
GPU                    NVIDIA A100-SXM4-40GB, 40507MiB
GPU count              1
CUDA                   12.6

numpy                  ✅ 2.0.2>=1.23.0
matplotlib             ✅ 3.10.0>=3.3.0
opencv-python          ✅ 4.12.0.88>=4.6.0
pillow                 ✅ 11.3.0>=7.1.2
pyyaml                 ✅ 6.0.3>=5.3.1
requests               ✅ 2.32.4>=2.23.0
scipy                  ✅ 1.16.3>=1.4.1
torch                  ✅ 2.8.0+cu126>=1.8.0
torch                  ✅ 2.8.0+cu126!=2.4.0,>=1.8.0; sys_platform == "win32"
torchvision            ✅ 0.23.0+cu126>=0.9.0
psutil                 ✅ 5.9.5
polars                 ✅ 1.25.2
ultralytics-thop       ✅ 2.0.18>=2.0.18
"""

from ultralytics import YOLO

from IPython.display import display, Image
#!yolo task=detect data=/content/drive/MyDrive/datasets/otc/data.yaml  mode=train model=yolo11n.pt epochs=100 rect=True imgsz=[1088, 1920] save_period=1

"""
Ultralytics 8.3.227 🚀 Python-3.12.12 torch-2.8.0+cu126 CUDA:0 (NVIDIA A100-SXM4-40GB, 40507MiB)
engine/trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/drive/MyDrive/datasets/otc/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=[1088, 1920], int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train2, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=True, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/detect/train2, save_frames=False, save_json=False, save_period=1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    430867  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
YOLO11n summary: 181 layers, 2,590,035 parameters, 2,590,019 gradients, 6.4 GFLOPs

Transferred 448/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
WARNING ⚠️ updating to 'imgsz=1920'. 'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'
train: Fast image access ✅ (ping: 0.3±0.0 ms, read: 465.1±174.7 MB/s, size: 1259.5 KB)
train: Scanning /content/drive/MyDrive/datasets/otc/labels/train.cache... 3598 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 3599/3599 40.4Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
val: Fast image access ✅ (ping: 0.8±0.8 ms, read: 203.1±144.5 MB/s, size: 1345.5 KB)
val: Scanning /content/drive/MyDrive/datasets/otc/labels/val.cache... 450 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 450/450 2.9Mit/s 0.0s
Plotting labels to /content/runs/detect/train2/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Image sizes 1920 train, 1920 val
Using 8 dataloader workers
Logging results to /content/runs/detect/train2
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      12.3G      1.441      1.588      1.367        136       1920: 100% ━━━━━━━━━━━━ 225/225 2.2it/s 1:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.3it/s 11.7s
                   all        450       6610        0.9      0.894      0.942      0.504

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      13.6G       1.28      1.043      1.246        119       1920: 100% ━━━━━━━━━━━━ 225/225 3.4it/s 1:06
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.5it/s 4.3s
                   all        450       6610      0.951      0.913      0.963      0.556

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      13.6G      1.241     0.8695      1.229        127       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.939      0.913      0.963      0.578

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      13.6G      1.202      0.814      1.213        120       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.934      0.874      0.944      0.541

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      13.6G       1.16      0.774      1.197        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.958       0.94      0.982      0.665

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      13.6G      1.084     0.7338      1.155        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.927       0.93      0.973       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      13.6G      1.036     0.6818      1.138        131       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.1s
                   all        450       6610      0.961      0.939      0.982      0.644

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      13.6G     0.9922     0.6536      1.113        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.965      0.949      0.983      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      13.6G     0.9264     0.6121      1.075        131       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.964      0.946      0.983      0.635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      13.6G     0.9161     0.5849       1.07        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.958      0.959      0.985      0.606

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      13.6G     0.9214     0.5743      1.076        125       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.963      0.945      0.981      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      13.6G     0.8539     0.5543      1.043        137       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.966      0.955      0.986      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      13.6G     0.8565      0.541      1.042        139       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.6s
                   all        450       6610      0.969      0.954      0.984      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      13.6G     0.8196     0.5203      1.026        136       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.973      0.963      0.987      0.745

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      13.6G     0.8287     0.5171      1.033        120       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.969      0.949      0.985       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      13.6G     0.8043     0.5106      1.024        120       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.958      0.962      0.983      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      13.6G     0.7741     0.4942      1.006        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.6it/s 4.1s
                   all        450       6610      0.968      0.959      0.984      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      13.6G     0.7719     0.4918      1.005        137       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.973      0.963      0.987      0.739

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      13.6G     0.7901     0.4886      1.013        140       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.4it/s 4.4s
                   all        450       6610      0.966      0.958      0.986      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      13.6G     0.7771     0.4814       1.01        137       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.957      0.944      0.981      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      13.6G     0.7443     0.4716     0.9923        123       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610       0.97      0.964      0.987      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100      13.6G      0.742     0.4671     0.9934        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.959      0.958      0.983      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      13.6G     0.7471     0.4608     0.9926        133       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.968       0.94      0.984      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      13.6G      0.767     0.4689      1.004        124       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.966      0.954      0.988      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      13.6G     0.7489     0.4576      0.995        124       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.1s
                   all        450       6610      0.964      0.962      0.986      0.744

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      13.6G     0.6995      0.442     0.9729        131       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.951      0.946      0.982      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      13.6G     0.6842     0.4331      0.964        139       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.963      0.954      0.988      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      13.6G      0.749     0.4458      0.995        121       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.964       0.95      0.982      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      13.6G     0.7117     0.4342     0.9765        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.1s
                   all        450       6610      0.965      0.951      0.986      0.676

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      13.6G     0.6642     0.4246     0.9547        130       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.959       0.96      0.987      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      13.6G     0.6905     0.4227     0.9664        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.971      0.951      0.987      0.788

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      13.6G     0.6826       0.42      0.964        122       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.971       0.96      0.988      0.763

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      13.6G      0.691      0.426     0.9694        132       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.2it/s 3.6s
                   all        450       6610      0.967       0.96      0.986      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      13.6G     0.6541     0.4094     0.9513        119       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.966      0.962      0.987      0.768

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      13.6G     0.6566     0.4074     0.9477        125       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.971      0.965      0.987      0.784

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      13.6G       0.64     0.3977     0.9435        132       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.978      0.958      0.989      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      13.6G     0.6395     0.4026     0.9428        138       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.976      0.962       0.99      0.782

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      13.6G     0.6363     0.3949     0.9419        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.973      0.967      0.989      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      13.6G     0.6741     0.4004     0.9591        127       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.976      0.966      0.991      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      13.6G     0.7207     0.4121      0.989        128       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.979      0.963      0.991      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      13.6G     0.6405     0.3932     0.9486        136       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.2it/s 3.6s
                   all        450       6610      0.976      0.967      0.991      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      13.6G     0.5881     0.3714     0.9238        131       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.977      0.965      0.991      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      13.6G     0.6007     0.3798     0.9298        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.971      0.971      0.991      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      13.6G     0.5878     0.3737     0.9233        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.2it/s 3.6s
                   all        450       6610      0.975      0.968       0.99        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      13.6G     0.6078     0.3763     0.9306        128       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.978      0.958       0.99      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      13.6G     0.6533     0.3902     0.9547        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.973      0.965       0.99      0.774

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      13.6G     0.6413      0.386     0.9469        139       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.978      0.967      0.992      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      13.6G     0.6005     0.3725     0.9303        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.969      0.975      0.991      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      13.6G     0.5643     0.3607     0.9106        141       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.968      0.974      0.991      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      13.6G     0.5621     0.3642     0.9132        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610       0.97       0.97      0.991       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      13.6G     0.5558     0.3558     0.9083        130       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.2it/s 3.5s
                   all        450       6610      0.976      0.966      0.991      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      13.6G     0.5594     0.3566     0.9109        123       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.6it/s 4.2s
                   all        450       6610      0.968      0.965       0.99      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      13.6G     0.5575     0.3565     0.9085        133       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610       0.97      0.959      0.988      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      13.6G     0.5644     0.3557     0.9118        130       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.975      0.964       0.99      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      13.6G     0.6017     0.3616     0.9286        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.977      0.961       0.99      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      13.6G     0.5655     0.3507     0.9114        133       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.971      0.967       0.99      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      13.6G     0.5441     0.3442     0.9041        133       1920: 100% ━━━━━━━━━━━━ 225/225 3.0it/s 1:15
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.976      0.963      0.991      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      13.6G     0.5397     0.3422     0.9023        121       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.973       0.96       0.99      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      13.6G     0.5918     0.3548     0.9205        124       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.978      0.967      0.991      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      13.6G     0.5701      0.349     0.9127        142       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.977      0.958       0.99      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      13.6G     0.5831     0.3541     0.9171        126       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.6it/s 4.2s
                   all        450       6610      0.965      0.967       0.99       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      13.6G     0.5769     0.3481     0.9158        127       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.978      0.965      0.991      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      13.6G     0.5789     0.3496     0.9172        125       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.973      0.962       0.99      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      13.6G     0.5534     0.3402     0.9099        141       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610       0.97      0.967      0.991      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      13.6G     0.5338     0.3341     0.8993        128       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.976      0.964      0.991      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100      13.6G     0.5425     0.3351      0.904        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.973      0.971      0.991      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      13.6G     0.5568     0.3378     0.9078        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.974      0.969      0.991      0.823

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      13.6G     0.5734     0.3404      0.911        116       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.978       0.96       0.99      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      13.6G     0.5641     0.3408     0.9102        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.975      0.961      0.991      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      13.6G     0.5677     0.3422     0.9166        136       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.7s
                   all        450       6610      0.973      0.968      0.991       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      13.6G     0.5732      0.342     0.9152        137       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.972      0.963      0.991      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      13.6G     0.5537     0.3373     0.9094        127       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.976      0.963      0.991       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      13.6G     0.5106     0.3247     0.8935        133       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.975      0.967      0.991      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      13.6G     0.5168     0.3273      0.896        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.979      0.963      0.991       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      13.6G     0.5396     0.3306     0.9008        123       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.2it/s 3.6s
                   all        450       6610      0.974      0.966      0.991      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      13.6G     0.5444     0.3287      0.904        136       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.974      0.965      0.991      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      13.6G     0.5357     0.3266      0.902        140       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.974      0.966      0.992      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100      13.6G      0.507     0.3178     0.8919        115       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610       0.97      0.966      0.992      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      13.6G      0.504     0.3164     0.8895        136       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.965      0.967      0.991       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      13.6G     0.5055      0.313     0.8891        129       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.1s
                   all        450       6610      0.971      0.965      0.991      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      13.6G     0.5089     0.3147     0.8904        138       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.981      0.959      0.991      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      13.6G     0.5046     0.3119     0.8914        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.6s
                   all        450       6610      0.976      0.964      0.991      0.829

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      13.6G       0.48     0.3053     0.8811        126       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.6it/s 4.1s
                   all        450       6610      0.971      0.966      0.991      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      13.6G     0.4718     0.3027     0.8774        128       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.976      0.965      0.991      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      13.6G     0.4872     0.3081     0.8843        134       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.0it/s 3.8s
                   all        450       6610      0.972      0.967      0.991      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100      13.6G     0.5115     0.3103     0.8936        139       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.1s
                   all        450       6610      0.977      0.965      0.991      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      13.6G     0.4912     0.3039     0.8854        126       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.3it/s 3.5s
                   all        450       6610      0.971       0.97      0.991      0.837

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      13.6G     0.4537     0.2931     0.8704        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.975       0.97      0.992      0.839

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      13.6G     0.4595     0.2958     0.8728        126       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.975       0.97      0.992      0.837

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      13.6G     0.4757     0.3004     0.8789        138       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.976      0.967      0.992      0.833
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      13.6G     0.4662     0.2954     0.8746        127       1920: 100% ━━━━━━━━━━━━ 225/225 2.9it/s 1:16
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 4.1it/s 3.7s
                   all        450       6610      0.974      0.972      0.992      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      13.6G     0.4584     0.2933     0.8709        135       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:14
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.977      0.967      0.992      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      13.6G     0.4423     0.2865     0.8649        124       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.974      0.968      0.991      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      13.6G     0.4339     0.2848     0.8636        136       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.9s
                   all        450       6610      0.974      0.969      0.992      0.836

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      13.6G     0.4347     0.2849      0.865        113       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.0s
                   all        450       6610      0.972      0.972      0.992      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      13.6G      0.436     0.2865     0.8653        122       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 3.9s
                   all        450       6610      0.976      0.968      0.992      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      13.6G     0.4314     0.2831     0.8624        130       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.7it/s 4.1s
                   all        450       6610      0.973      0.971      0.992      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100      13.6G     0.4305     0.2833     0.8644        131       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.976      0.969      0.992      0.838

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      13.6G      0.426     0.2781     0.8624        121       1920: 100% ━━━━━━━━━━━━ 225/225 3.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.9it/s 3.8s
                   all        450       6610      0.972      0.971      0.992      0.837

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      13.6G     0.4224     0.2779      0.859        126       1920: 100% ━━━━━━━━━━━━ 225/225 3.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 3.8it/s 4.0s
                   all        450       6610      0.973       0.97      0.992      0.839

100 epochs completed in 2.133 hours.
Optimizer stripped from /content/runs/detect/train2/weights/last.pt, 5.7MB
Optimizer stripped from /content/runs/detect/train2/weights/best.pt, 5.7MB

Validating /content/runs/detect/train2/weights/best.pt...
Ultralytics 8.3.227 🚀 Python-3.12.12 torch-2.8.0+cu126 CUDA:0 (NVIDIA A100-SXM4-40GB, 40507MiB)
YOLO11n summary (fused): 100 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 2.4it/s 6.2s
                   all        450       6610      0.973      0.971      0.992      0.839
Speed: 0.6ms preprocess, 3.3ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to /content/runs/detect/train2
💡 Learn more at https://docs.ultralytics.com/modes/train
"""
