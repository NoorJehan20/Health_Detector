## 💡AI-Powered Heart Health Risk Detector

This project is an AI-driven heart disease risk prediction system using machine learning models. It analyzes clinical and demographic data to predict the likelihood of heart disease, assisting early diagnosis and preventive healthcare.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## Project Overview

The goal is to build predictive models to classify if a patient has heart disease (`condition` column). The project involves:

- Data exploration and visualization
- Feature engineering and preprocessing
- Model training using Logistic Regression, Random Forest, and SVM
- Hyperparameter tuning with GridSearchCV
- Model evaluation with accuracy, classification report, confusion matrix, and ROC curves
- Saving the best performing model for deployment

---

## Dataset

The dataset used is the **Heart Disease Cleveland Dataset** (commonly known as `heart_cleveland_upload.csv`) which includes attributes such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise induced angina
- ST depression induced by exercise relative to rest
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia
- Condition (target variable: 0 = no heart disease, 1 = heart disease)

---

## Features

- Data visualization using histograms, KDE plots, violin plots, and countplots.
- Correlation heatmap with clustering to explore feature relationships.
- Preprocessing includes renaming columns for clarity and one-hot encoding categorical variables.
- Stratified train-test split to maintain class distribution.

---

## Installation

Clone the repository:

```bash
git clone https:https://github.com/NoorJehan20/Health_Risk_Detector.git
cd Health-Risk-Detector
````

Install required Python packages (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```
---

## Usage

Run the Jupyter notebook or Python script to:

* Load and explore the dataset
* Train and tune models with GridSearchCV
* Evaluate models with metrics and visualization
* Save the best model (`best_model.pkl`)

---

## Modeling Approach

* Logistic Regression with standard scaling and tuning regularization parameter `C`
* Random Forest with tuning of number of estimators, max depth, and min samples split
* Support Vector Machine (SVM) with linear and RBF kernels and regularization tuning

---

## Evaluation

Model performance is evaluated using:

* Accuracy score on test data
* Classification report (precision, recall, F1-score)
* Confusion matrix visualization
* ROC curve and AUC score comparison

---

## Technologies Used

* Python
* Pandas & NumPy for data manipulation
* Matplotlib & Seaborn for visualization
* scikit-learn for modeling and evaluation
* joblib for model serialization

---

## Author

Noor Jehan
[GitHub Profile](https://github.com/noorjehan20)
[LinkedIn Profile](https://www.linkedin.com/in/noor-jehan-5a4161278/)
