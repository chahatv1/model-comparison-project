# 🔍 Machine Learning Model Comparison & Hyperparameter Tuning

This project demonstrates the training, evaluation, and optimization of multiple machine learning models using Python and scikit-learn. It focuses on comparing performance using key classification metrics and improving results through hyperparameter tuning (using GridSearchCV).

---

## 📂 Project Overview

We evaluate and compare the following classification models on the Breast Cancer dataset:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree

We also use **GridSearchCV** to find the best hyperparameters for Random Forest and visualize the results.

---

## 🛠️ Features

- 📊 Evaluation using **Accuracy**, **Precision**, **Recall**, and **F1-Score**
- 🔧 Hyperparameter tuning with **GridSearchCV**
- 📈 Visual comparison using **matplotlib** and **seaborn**
- 🧪 Dataset: Breast Cancer dataset from `sklearn.datasets`

---

## 🧠 What You’ll Learn

- How to train and evaluate multiple classification models
- How to compare model performance using proper metrics
- How to use GridSearchCV for model optimization
- How to visualize performance for better insights

---

## 📁 Files in this Repo

| File | Description |
|------|-------------|
| `main.py` | Main script for model training, evaluation, and tuning |
| `.gitignore` | Prevents virtual environment and cache files from being pushed |
| `README.md` | This file! Contains all the project info |

---

## 📊 Results (F1 Scores)

| Model                | F1 Score |
|----------------------|----------|
| Logistic Regression  | 0.9655   |
| Random Forest (Tuned)| **0.9722** ✅ |
| SVM                  | 0.9594   |
| Decision Tree        | 0.9577   |

---
