# yolo
yolov11

YOLOv11-PSConv+MAN-Faster+C2BRA Model
This is an advanced object detection model based on the YOLOv11 architecture, integrating three innovative modules: PSConv, MAN-Faster, and C2BRA. It significantly improves detection accuracy and inference efficiency. The project includes complete training, validation, and deployment code.

Project Structure 📂
├── yolo11-PSConv+MAN-Faster+C2BRA.yaml    # Main model configuration file
├── yolo11-PSConv+MAN-Faster.yaml          # Variant model configuration
├── yolo11-PSConv+c3k2_ETB.yaml            # Variant model configuration
├── yolo11-PSConv+c3k2_FDT.yaml            # Variant model configuration
├── train.py                               # Model training script
├── val.py                                 # Model validation script
├── track.py                               # Object tracking script
├── transform_PGI.py                       # Data augmentation transformation script
├── test_yaml.py                           # Configuration file testing script
├── test_env.py                            # Environment testing script
├── yolo11n.pt                             # Pretrained model weights
├── .gitignore                             # Git ignore configuration
└── CITATION.cff                           # Citation format file

Module Explanation 🧠
1.PSConv (Progressive Sparse Convolution)
  ·Progressive dilation rate design
  ·Sparse kernel combination
  ·Enhanced multi-scale feature capture capability
2.MAN-Faster (Multi-Axis Attention Faster)
  ·Dual-path parallel attention mechanism
  ·Channel-spatial joint attention
  ·Feature enhancement with zero additional computational cost
3.C2BRA (Cross-Stage Bottleneck Residual Attention)
  ·Cross-stage feature fusion
  ·Bottleneck residual design
  ·Adaptive feature recalibration
