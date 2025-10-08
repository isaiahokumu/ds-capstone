# Brain Tumor Detection & Intelligence System

An integrated deep learning project that detects brain tumors from MRI scans using **Convolutional Neural Networks (CNNs)**, **VGG16/VGG19 transfer learning**, and **Autoencoders** for anomaly detection — combined with **OCR extraction**, **real-time news sentiment analysis**, and **cloud-based deployment**.

---

## Project Overview

This project aims to build an end-to-end AI-powered system for **brain tumor detection, research awareness, and insight generation**.

### Key Components
- **Deep Learning:** Classification & segmentation of brain tumors using CNNs and transfer models (VGG16/VGG19).
- **Autoencoders:** Unsupervised anomaly detection to identify abnormal brain regions.
- **OCR:** Extract and interpret medical text from scanned reports or PDFs.
- **News & Sentiment Analysis:** Fetch latest medical news related to brain tumors and classify sentiment.
- **Cloud & Deployment:** Model training and deployment on cloud platforms (AWS, GCP, or Azure).
- **Web Interface:** Streamlit application for clinicians and researchers to visualize results.

---

## Project Architecture
```graphql
brain-tumor-ai/
│
├── data/
│   ├── raw/                 # Raw MRI data
│   ├── processed/           # Preprocessed/normalized data
│   └── reports/             # OCR input files
│
├── models/
│   ├── cnn_model.h5         # CNN baseline model
│   ├── vgg16_model.h5       # Transfer learning model (VGG16)
│   ├── vgg19_model.h5       # Transfer learning model (VGG19)
│   └── autoencoder_model.h5 # Autoencoder anomaly detection
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_cnn_training.ipynb
│   ├── 03_vgg16_training.ipynb
│   ├── 04_autoencoder.ipynb
│   ├── 05_gradcam_visualization.ipynb
│   └── 06_sentiment_analysis.ipynb
│
├── src/
│   ├── data_loader.py           # Handles data ingestion and augmentation
│   ├── model_cnn.py             # CNN architecture
│   ├── model_vgg16.py           # Transfer learning model
│   ├── model_autoencoder.py     # Autoencoder model
│   ├── gradcam_utils.py         # Explainability (Grad-CAM)
│   ├── sentiment_scraper.py     # Fetch & analyze news sentiment
│   ├── ocr_extractor.py         # OCR extraction using Tesseract
│   ├── api.py                   # FastAPI model inference endpoint
│   └── streamlit_app.py         # User interface
│
├── deployment/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── cloud_setup.md
│
├── tests/
│   ├── test_models.py
│   ├── test_ocr.py
│   └── test_api.py
│
├── README.md
└── LICENSE
```

---

## Features

| Component | Description |
|------------|-------------|
| **CNN Classifier** | Custom convolutional model for baseline classification |
| **Transfer Learning (VGG16/VGG19)** | Pre-trained models fine-tuned for tumor detection |
| **Autoencoder** | Unsupervised anomaly detection using reconstruction errors |
| **OCR Extraction** | Extracts medical text from reports/images using Tesseract |
| **Web Scraping & APIs** | Collects latest brain tumor research/news |
| **Sentiment Analysis** | Analyzes tone and emotion in medical news |
| **Cloud Deployment** | Model hosting via AWS SageMaker, GCP Vertex, or Streamlit Cloud |
| **Explainability (Grad-CAM)** | Visual heatmaps showing regions influencing predictions |

---

## Tech Stack

**Machine Learning**
- TensorFlow / Keras  
- Scikit-learn  
- NumPy / Pandas  
- OpenCV  

**NLP & OCR**
- Tesseract (OCR)  
- Hugging Face Transformers  
- BeautifulSoup / Requests  

**Web & API**
- Streamlit  
- FastAPI  
- Docker  

**Cloud**
- AWS SageMaker / GCP Vertex AI  
- Streamlit Cloud  

---

## Setup Instructions

### Clone the Repository

git clone https://github.com/<your-username>/brain-tumor-ai.git
cd brain-tumor-ai


### Install Dependencies
pip install -r deployment/requirements.txt

### Download Dataset

Use the Kaggle Brain MRI Dataset

Unzip it into:
```bash
/data/raw/
```

### Train Models

Run notebooks in sequence:
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

### Launch FastAPI Service
```bash
uvicorn src.api:app --reload
```
### Docker Build (Optional)
```bash
docker build -t brain-tumor-ai .
docker run -p 8501:8501 brain-tumor-ai
```
### Cloud Deployment
Use GCP Vertex AI for training and prediction.

| Metric               | CNN | VGG16 | VGG19 | Autoencoder |
| -------------------- | --- | ----- | ----- | ----------- |
| Accuracy             | TBD | TBD   | TBD   | -           |
| Recall (Sensitivity) | TBD | TBD   | TBD   | -           |
| F1 Score             | TBD | TBD   | TBD   | -           |
| AUC                  | TBD | TBD   | TBD   | -           |


