# Employee Retention Prediction Project
### Overview
This project focuses on predicting employee retention at a company using machine learning models. The project utilizes two main types of models: Logistic Regression and Tree-Based Machine Learning (Decision Tree and Random Forest). The goal is to identify factors influencing employee retention and propose actionable business recommendations to reduce turnover.
### Understand the business scenario and problem

The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?

Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company. 

If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.
## Project Structure
The project follows the PACE (Plan, Analyze, Construct, Execute) framework, ensuring that every phase of the machine learning pipeline aligns with clear business objectives and stakeholder needs.

## Files
employee_retention_analysis.ipynb: Jupyter notebook containing the code for data analysis, model construction, evaluation, and insights.
data/: Folder containing the dataset used for the analysis.
README.md: Project documentation (this file).
## PACE Framework

**Plan**
In the planning stage, we identified the business objective of reducing employee turnover by predicting which employees are most likely to leave the company. This was based on various features, including satisfaction scores, evaluation scores, project counts, and work hours.

**Analyze**
We performed exploratory data analysis to identify patterns and correlations in the dataset. We observed that employee satisfaction and last evaluation scores had a strong influence on whether employees left the company.

```python

# Exploratory Data Analysis (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/employee_data.csv')

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Employee Retention Data')
plt.show()
```
**Construct**
During this stage, we built two machine learning models:

**Logistic Regression:** A simple yet effective baseline model.
Tree-Based Models (Decision Tree and Random Forest): More complex models that improved prediction performance.
```python

# Logistic Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split the data
X = df.drop(columns=['left_company'])
y = df['left_company']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Train Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred_tree = tree.predict(X_test)
tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
print(f"Decision Tree AUC: {tree_auc}")

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest AUC: {rf_auc}")
```
**Execute**
In the Execute stage, we interpreted the model results and translated them into actionable business insights. The following recommendations were made to improve employee retention:

**Cap Projects:** Limit the number of projects assigned to employees to avoid burnout.
**Promotions for Tenured Employees:** Investigate why employees with four or more years of service show higher dissatisfaction and consider promoting or rewarding them.
**Overtime Policies:** Reward employees who work longer hours, or adjust expectations to reduce the need for excessive overtime.
**Clarify Policies:** Ensure employees understand the company’s policies on overtime and workload.
**Company Culture:** Encourage discussions within teams to address work culture issues and ensure that high evaluation scores are not biased toward employees working over 200 hours per month.
```python

# Feature Importance from Random Forest Model
import numpy as np
importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importance - Random Forest')
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.show()
```
## Model Results Summary
**Logistic Regression:**

Precision: 80%
Recall: 83%
F1-score: 80%
Accuracy: 83%

**Tree-Based Models:**

**Decision Tree:** Accuracy of 96.2%, AUC of 93.8%, F1-score of 88.7%
**Random Forest:** Modestly outperformed Decision Tree.

## Recommendations for Future Work
**Remove Potential Data Leakage:** Re-run models without last_evaluation to test for overfitting and data leakage.
**Focus on Satisfaction and Performance:** Rather than solely predicting retention, try to predict the satisfaction_score or performance_score to provide more targeted business insights.
**Clustering:** Apply a K-Means clustering algorithm to analyze employee groups, which may yield additional insights on retention and dissatisfaction factors.

## Conclusion
The project confirmed that employees at the company are overworked, and satisfaction and evaluation scores play key roles in retention decisions. By addressing the underlying causes of dissatisfaction and overwork, the company can improve retention rates and reduce turnover.