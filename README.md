# CNN News Article Classification Using Hierarchical LLM-Assisted Labels

**Course:** ENGR 5775: Data Mining and Knowledge Discovery  
**Professor:** Dr. Masoud Makrehchi  
**Group:** Group 1  

## Group Members

- Jason Stuckless
- Paria Sarzaeim

---

# Project Overview

This project investigates whether a locally hosted Large Language Model (LLM) can generate high-quality labels suitable for supervised news article classification.

Unlike a traditional flat classification approach, the project uses **hierarchical LLM-assisted labeling**. The LLM first predicts a broad **Category** for each article and then predicts a **Section** constrained by that Category. The resulting labels are used to train multiple machine learning classifiers and are compared with classifiers trained using the original human-generated CNN labels.

The project evaluates whether LLM-generated labels can serve as a practical substitute for manually annotated training data.

---

# Objectives

The project consists of four primary stages.

1. Prepare and split the CNN dataset into training and testing sets.
2. Generate hierarchical Category and Section labels using a locally hosted LLM.
3. Train machine learning classifiers using both:
   - Original CNN labels
   - LLM-generated labels
4. Compare classifier performance using standard evaluation metrics.

---

# Dataset

The project uses the CNN News Articles dataset.

Each article contains two target labels:

- Category
- Section

The original labels are treated as ground truth throughout the project.

---

# Hierarchical Labeling Strategy

The original CNN dataset follows a hierarchical structure.

```
Category
│
├── business
│   ├── business
│   ├── business-food
│   ├── business-money
│   ├── cars
│   ├── economy
│   ├── energy
│   ├── homes
│   ├── investing
│   ├── media
│   ├── perspectives
│   ├── success
│   └── tech
│
├── entertainment
│   ├── entertainment
│   ├── celebrities
│   └── movies
│
├── health
│   └── health
│
├── news
│   ├── africa
│   ├── americas
│   ├── asia
│   ├── australia
│   ├── china
│   ├── europe
│   ├── india
│   ├── intl_world
│   ├── living
│   ├── middleeast
│   ├── opinions
│   ├── uk
│   ├── us
│   ├── weather
│   └── world
│
├── politics
│   └── politics
│
└── sport
    ├── sport
    ├── football
    ├── golf
    ├── motorsport
    └── tennis
```

The hierarchy is automatically extracted from the original training dataset and is used to constrain LLM predictions.

---

# LLM-Assisted Label Generation

Label generation is performed in two stages.

## Stage 1 – Category Prediction

The LLM predicts exactly one Category from:

- news
- business
- health
- entertainment
- sport
- politics

## Stage 2 – Section Prediction

Once the Category has been selected, the LLM predicts a Section belonging only to that Category.

For example, if the predicted Category is **sport**, the available Sections are limited to:

- sport
- football
- golf
- motorsport
- tennis

This hierarchical approach reduces the search space, prevents invalid Category–Section combinations, and more closely mirrors the editorial structure of the original CNN dataset.

---

# Machine Learning Models

Three classical machine learning classifiers are trained.

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes

Each classifier is trained twice.

### Experiment 1

Training labels:

- Original CNN labels

### Experiment 2

Training labels:

- Hierarchically generated LLM labels

All models are evaluated using the same original testing dataset.

---

# Feature Extraction

Article text is converted into numerical feature vectors using TF-IDF vectorization.

The same TF-IDF vectorizer is used for both experiments. Only the source of the training labels changes, ensuring a fair comparison between original and LLM-generated labels.

---

# Evaluation

Classifier performance is evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrices

The project also evaluates the agreement between:

- Original and LLM-generated Categories
- Original and LLM-generated Sections

---

# Project Structure

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
│   ├── category_classification_prompt.txt
│   └── section_classification_prompt.txt
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
│   │   ├── ollama_client.py
│   │   └── validate_labels.py
│   │
│   └── utils/
│       ├── io.py
│       └── logging.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Pipeline

The complete project consists of seven stages.

1. Prepare the dataset.
2. Generate hierarchical LLM labels.
3. Validate all datasets and hierarchical label pairs.
4. Vectorize article text using TF-IDF.
5. Train the classification models.
6. Evaluate classifier performance.
7. Generate figures and summary tables.

Run the complete pipeline using:

```bash
python src/run_pipeline.py
```

Alternatively, each stage may be executed independently.

---

# Software Requirements

- Python 3.11
- Ollama
- Llama 3.2 (3B) or another compatible Ollama model

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

# Experimental Design

The project compares two supervised learning conditions.

## Experiment 1

Training Labels

- Original CNN labels

Testing Labels

- Original CNN labels

## Experiment 2

Training Labels

- Hierarchically generated LLM labels

Testing Labels

- Original CNN labels

Using the same testing dataset for both experiments ensures that classifier performance differences are attributable only to the source of the training labels.

---

# Outputs

The project automatically generates:

- Hierarchically LLM-labeled training dataset
- Hierarchical labeling audit
- TF-IDF feature matrices
- Trained machine learning models
- Evaluation metrics
- Confusion matrices
- Performance comparison figures
- Summary tables

---

# Future Work

Potential future improvements include:

- Improving prompt engineering for hierarchical label generation.
- Investigating larger LLMs for label generation.
- Exploring transformer-based classifiers (e.g., BERT) for semantic text classification.
- Comparing additional machine learning classifiers.
- Hyperparameter optimization for all classification models.
- Evaluating alternative hierarchical prompting strategies.