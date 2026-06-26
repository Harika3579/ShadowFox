# 🏦 Task 2: Loan Approval Prediction

## 📌 Project Overview
This project is part of the **ShadowFox AIML Internship** (Phase 3 - Intermediate Task). The objective is to build an intelligent predictive model capable of determining whether a loan application will be approved or denied based on the applicant's financial and demographic history.

## 📊 Dataset
The dataset consists of historical loan application records, featuring attributes such as:
* **ApplicantIncome & CoapplicantIncome:** Financial standing of the borrowers.
* **Credit_History:** A binary indicator of previous credit compliance.
* **Dependents, Education, & Employment:** Demographic contextual factors.
* **Loan_Status:** The target variable (Y for Approved, N for Denied).

## 🛠️ Methodology
1. **Exploratory Data Analysis (EDA):** Leveraged `Plotly` to visualize the distribution of loan statuses and identify correlations, particularly the heavy weight of `Credit_History` on approval rates.
2. **Data Preprocessing:** * Handled missing values using statistical imputation (Mode for categorical, Median for numerical).
   * Applied **One-Hot Encoding** to transform categorical variables into a machine-readable format.
3. **Feature Scaling:** Applied `StandardScaler` to normalize income and loan amounts, ensuring distance-based algorithms performed optimally.
4. **Model Selection:** Trained and evaluated multiple algorithms:
   * Random Forest Classifier
   * Support Vector Classifier (SVC)
5. **Hyperparameter Tuning:** Applied `GridSearchCV` to the Random Forest model to optimize estimators and depth, while implementing `class_weight='balanced'` to address the dataset's inherent class imbalance.

## 📈 Results & Analysis
The models successfully established a baseline, but highlighted a classic real-world data limitation:

* **Tuned Random Forest Accuracy:** 77%
* **SVC Accuracy:** 76%
* **The Imbalance Challenge:** While the models easily identified Approved loans (Class 1 Recall: 96%), they struggled to accurately isolate Denied loans (Class 0 Recall: 42%). Even with balanced class weights, the dataset's limited size and overlapping feature boundaries prevented higher recall for the minority class. In a production environment, implementing synthetic data generation (SMOTE) would be the recommended next step.

## 🔗 Proof of Work
* **Task screenshots: https://drive.google.com/drive/folders/1m2MtJqwAlDvwNQ-oaMgEftDHqjSBhd8-?usp=sharing
