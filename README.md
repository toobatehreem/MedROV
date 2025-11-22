# MedROV

Official PyTorch implementation for paper "**MedROV**: Towards Real-Time Open-Vocabulary Detection Across Diverse Medical Imaging Modalities", accepted at WACV 2026. [Paper Link](http://)

## Abstract

Traditional object detection models in medical imaging operate within a closed-set paradigm, limiting their ability to detect objects of novel labels. Open-vocabulary object detection (OVOD) addresses this limitation but remains underexplored in medical imaging due to dataset scarcity and weak text-image alignment. To bridge this gap, we introduce MedROV, the first Real-time Open Vocabulary detection model for medical imaging. To enable open-vocabulary learning, we curate a large-scale dataset, Omnis, with 600K detection samples across nine imaging modalities and introduce a pseudo-labeling strategy to handle missing annotations from multi-source datasets. Additionally, we enhance generalization by incorporating knowledge from a large pre-trained foundation model. By leveraging contrastive learning and cross-modal representations, MedROV effectively detects both known and novel structures. Experimental results demonstrate that MedROV outperforms the previous state-of-the-art foundation model for medical image detection with an average absolute improvement of 40 mAP50, and surpasses closed-set detectors by more than 3 mAP50, while running at 70 FPS, setting a new benchmark in medical detection.

## Model Architecture
<img width="1413" height="872" alt="image" src="https://github.com/user-attachments/assets/7eaa1e2b-10cc-4c94-b90a-f9c6abb2a0cb" />

## Getting Started

### Installation

To set up the environment and install the required packages, run the following commands:

```bash
conda create -n medrov python=3.10
conda activate medrov
git clone https://github.com/toobatehreem/MedROV
cd MedROV
pip install -e .
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib
```

### Training

To train the MedROV model, use the following code snippet:

```python
from ultralytics import YOLOWorld

# Load the model configuration and weights
model = YOLOWorld("yolov8l-worldv2.pt")

# Start training
results = model.train(data="data.yaml", epochs=20, batch=64, optimizer='AdamW', lr0=0.0002, weight_decay=0.05, device=(0,1,2,3))

```

### Testing

To evaluate the trained model, you can use the following code:

```python
from ultralytics import YOLOWorld

# Load the model configuration and weights
model = YOLOWorld("/MedROV/checkpoints/medrov.pt")

# Validate the model
model.val(data="data.yaml", device=0)

# Print evaluation metrics
print(f"Mean Average Precision @ .5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")
```


## Acknowledgements

We sincerely thank [Ultralytics](https://github.com/ultralytics/ultralytics) for providing the YOLOWorld code.

