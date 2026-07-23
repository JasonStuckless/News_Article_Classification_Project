# CNN News Article Classification Using LLM-Assisted Labels

**Course:** ENGR 5775: Data Mining and Knowledge Discovery  
**Professor:** Dr. Masoud Makrehchi  
**Group:** Group 1  

**Group Members**
- Jason Stuckless
- Paria Sarzaeim

---

## Project Overview

This project investigates the use of a Large Language Model (LLM) to generate labels for a news article classification task and compares its effectiveness against using the original human-assigned labels.

The project uses the **CNN News Articles** dataset and predicts two target variables:

- **Category**
- **Section**

The primary objective is to determine how classifier performance changes when trained using LLM-generated labels instead of the original dataset labels.

---

## Objectives

The project consists of four primary stages:

1. Prepare and split the CNN dataset into training and testing sets.
2. Use a local LLM (via Ollama) to generate Category and Section labels for every article in the training dataset.
3. Train multiple machine learning classifiers using both:
   - Original CNN labels
   - LLM-generated labels
4. Compare the resulting models using standard classification metrics.

---

## Dataset

The project uses the CNN News Articles dataset.

The original dataset contains human-assigned labels for:

- Category
- Section

These labels serve as the ground truth for evaluating both the LLM-generated labels and the trained classifiers.

---

## LLM-Assisted Labeling

A locally hosted LLM is used to generate labels for each training article.

The model is instructed to:

- predict exactly one Category
- predict exactly one Section
- return only valid JSON
- choose labels exclusively from the predefined label lists

The generated labels are stored separately from the original dataset to preserve the original ground truth.

---

## Machine Learning Models

Three classical text classification models are trained.

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes

Each classifier is trained twice:

1. Using the original dataset labels.
2. Using the LLM-generated labels.

This produces a total of twelve trained models.

---

## Feature Extraction

Articles are converted into numerical feature vectors using TF-IDF vectorization.

The same vectorizer is used for both experiments to ensure that the only experimental difference is the source of the labels.

---

## Evaluation

All models are evaluated using the original testing dataset.

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrices

The project also measures the agreement between the original labels and the LLM-generated labels.

---

## Project Structure

```
CNN_Project/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── logs/
│
├── models/
│   ├── category/
│   └── section/
│
├── prompts/
│   └── article_classification_prompt.txt
│
├── results/
│   ├── confusion_matrices/
│   ├── figures/
│   ├── metrics/
│   └── tables/
│
├── src/
│   ├── run_pipeline.py
│   ├── 01_prepare_dataset.py
│   ├── 02_generate_llm_labels.py
│   ├── 03_validate_dataset.py
│   ├── 04_vectorize_dataset.py
│   ├── 05_train_models.py
│   ├── 06_evaluate_models.py
│   ├── 07_generate_figures.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   │   └── validate_labels.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── logging.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline

The project follows a seven-stage pipeline.

1. Prepare the dataset.
2. Generate LLM-assisted labels.
3. Validate all generated datasets.
4. Vectorize the articles using TF-IDF.
5. Train the classification models.
6. Evaluate model performance.
7. Generate figures and summary tables.

The complete pipeline can be executed using:

```bash
python src/run_pipeline.py
```

Alternatively, each stage may be executed independently.

---

## Software Requirements

- Python 3.11
- Ollama
- Local LLM (configured in `config/config.yaml`)

Python dependencies are listed in:

```
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Experimental Design

The project compares two training conditions.

### Experiment 1

Training labels:

- Original CNN labels

Testing labels:

- Original CNN labels

### Experiment 2

Training labels:

- LLM-generated labels

Testing labels:

- Original CNN labels

Using the same testing dataset for both experiments ensures that the comparison isolates the effect of replacing the training labels with LLM-generated labels.

---

## Outputs

The project automatically generates:

- LLM-labeled training dataset
- Validation reports
- TF-IDF feature matrices
- Trained classification models
- Evaluation metrics
- Confusion matrices
- Performance comparison figures
- Summary tables
