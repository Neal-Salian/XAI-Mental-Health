# Mental Health Text Classification using DistilBERT

This project implements a Natural Language Processing (NLP) model to classify mental health-related posts from Reddit into five distinct categories. Using the Hugging Face `transformers` library, we fine-tune a **DistilBERT** model to achieve high-accuracy classification.

## 📌 Overview

Mental health monitoring on social media is a growing field of research. This repository contains a pipeline to preprocess text data, train a transformer model, and evaluate its performance in distinguishing between various mental health conditions based on user-generated text.

## 📂 Dataset

The dataset used is the **Reddit Mental Health Data**, sourced from Kaggle. It consists of Reddit posts associated with specific mental health subreddits.

**Target Labels:**
The model classifies text into the following 5 categories:
* `0`: **Stress**
* `1`: **Depression**
* `2`: **Bipolar disorder**
* `3`: **Personality disorder**
* `4`: **Anxiety**

**Preprocessing:**
* The `title` and `text` columns are concatenated to form a `text_full` feature for richer context.
* Data is split into Training (80%) and Testing (20%) sets with stratification.

## 🛠️ Tech Stack

* **Python 3.10+**
* **PyTorch** (Deep Learning Framework)
* **Hugging Face Transformers** (Model & Tokenizer)
* **Scikit-Learn** (Metrics & Split)
* **Pandas & NumPy** (Data Manipulation)
* **Seaborn & Matplotlib** (Visualization)

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/mental-health-classification.git](https://github.com/your-username/mental-health-classification.git)
   cd mental-health-classification

   pip install torch torchvision transformers pandas numpy scikit-learn matplotlib seaborn

   🚀 Model & Training
Base Model: distilbert-base-uncased

Task: Sequence Classification (Multi-class)

Hyperparameters:

Epochs: 3

Batch Size: 16

Learning Rate: 2e-5

Weight Decay: 0.01

The model uses the Trainer API from Hugging Face for optimized training loops and evaluation.

📊 Results
The model achieved an overall Accuracy of 80% on the test set.
Class,Precision,Recall,F1-Score,Support

Stress,0.84,0.86,0.85,236

Depression,0.74,0.73,0.74,241

Bipolar,0.88,0.81,0.84,237

Personality,0.77,0.78,0.78,240

Anxiety,0.78,0.83,0.81,238

Confusion Matrix: The model shows strong separation between classes, with slight confusion primarily between Anxiety and Stress, which are semantically similar.

💻 Usage
Training

Run the Jupyter Notebook Mental_health_Classification.ipynb to download the dataset, preprocess data, and train the model.

Inference Example

To use the saved model for prediction:
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

# Load Model
model_path = "./mental_health_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Define Labels
labels = ['Stress', 'Depression', 'Bipolar', 'Personality', 'Anxiety']

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get predicted class index
    pred_idx = np.argmax(logits.numpy(), axis=1)[0]
    return labels[pred_idx]

# Test
sample_text = "I feel extremely anxious and panicked about my upcoming exams."
print(f"Prediction: {predict(sample_text)}")

 
# ⚠️Disclaimer
This model is a research project and not a diagnostic tool. It is trained on social media data and should not be used for medical diagnosis. If you or someone you know is struggling with mental health, please seek professional help.


# 📜License
This project is open-source. Please check the dataset license on Kaggle regarding the usage of Reddit data.
