# 🏡 Task 1: Boston House Price Prediction

## 📌 Project Overview
This project is part of the **ShadowFox AIML Internship** (Phase 2 - Beginner Task). The objective is to design and implement a robust machine learning regression model to accurately predict the median value of owner-occupied homes in Boston (`MEDV`) based on various attributes like crime rates, number of rooms, and property age.

## 📊 Dataset
The dataset utilized is the classic **Boston Housing Dataset**, which contains features such as:
* **CRIM:** Per capita crime rate by town
* **RM:** Average number of rooms per dwelling
* **AGE:** Proportion of owner-occupied units built prior to 1940
* **MEDV:** Median value of owner-occupied homes in $1000s (Target Variable)

## 🛠️ Methodology
The solution pipeline involves the following key steps:
1. **Data Preprocessing:** * Handled missing values by imputing column medians.
   * Addressed extreme outliers in the target variable using the **Interquartile Range (IQR)** technique to ensure a robust model.
2. **Data Splitting:** Separated the dataset into training (80%) and testing (20%) sets.
3. **Model Selection & Training:** Evaluated multiple algorithms to find the best fit:
   * Linear Regression
   * Decision Tree Regressor
   * Gradient Boosting Regressor
4. **Fine-Tuning:** Applied `GridSearchCV` to optimize hyperparameters (learning rate, max depth, estimators) for the top-performing model.

## 📈 Results & Evaluation
The models were evaluated based on **Mean Squared Error (MSE)** and **R-squared ($R^2$) Score**. 

* **Best Base Model:** Gradient Boosting Regressor
* **Performance:** Achieved an $R^2$ score of **0.85**, demonstrating that the model can explain 85% of the variance in Boston house prices. 
* *Note on Fine-Tuning:* While Grid Search was utilized to find optimal cross-validated parameters (`learning_rate=0.2`, `max_depth=3`, `n_estimators=100`), the base default parameters generalized slightly better to the unseen test data.

## 🔗 Proof of Work
* **Task Screenshots: https://drive.google.com/drive/folders/1-CXgP9hbKQMm8MAVCTkp39xe16A2ZKZQ?usp=sharing
