# PCB Defect Detection using YOLOv5

## Overview

This project implements an automated Printed Circuit Board (PCB) defect detection system using the YOLOv5 object detection framework. The model is trained to identify multiple PCB manufacturing defects, reducing the need for manual inspection and improving quality assurance.

---

## Features

- YOLOv5-based object detection
- Multi-class PCB defect classification
- Automatic annotation conversion (XML → YOLO format)
- Dataset preprocessing
- Training and inference notebooks
- Visualization of prediction results

---

## Technologies Used

- Python
- YOLOv5
- PyTorch
- OpenCV
- NumPy
- Jupyter Notebook

---

## Dataset

The project uses annotated PCB images containing common manufacturing defects.

Examples include:

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper

---

## Repository Structure

```
PCB-Defect-Detection/
│
├── notebooks/
├── PCB_DATASET/
├── XmlToTxt-master/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/ChatushRaj/PCB-Defect-Detection.git

cd PCB-Defect-Detection

pip install -r requirements.txt
```

---

## Workflow

1. Dataset Preparation
2. Annotation Conversion
3. YOLO Dataset Generation
4. Model Training
5. Prediction
6. Evaluation

---

## Future Improvements

- YOLOv8 Migration
- Real-time Camera Detection
- Model Deployment using Flask/FastAPI
- Edge Device Optimization

---

## Author

**Chatush Raj**

B.Tech Computer Science (AI & ML)

GitHub:
https://github.com/ChatushRaj
