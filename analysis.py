import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
df = pd.read_csv('heart.csv')

# initial look at the data
print(df.head())
print(df.info())

# checking gender distribution
# 1 = male, 0 = female 
gender_counts = df['sex'].value_counts()
print("\ngender distribution:")
print(f"males (1): {gender_counts[1]}")
print(f"females (0): {gender_counts[0]}")

# grouping data by sex to see the differences in averages
gender_grouped = df.groupby('sex').mean()
print("\naverages by gender:")
print(gender_grouped)

# analyzing chest pain type (cp) by gender
# cp types: 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic
plt.figure(figsize=(10, 6))
sns.countplot(x='cp', hue='sex', data=df, palette='pastel')
plt.title('chest pain types by gender')
plt.xlabel('chest pain type (0-3)')
plt.ylabel('count')
plt.legend(title='gender', labels=['female', 'male'])
plt.show()

# correlation for females
female_corr = df[df['sex'] == 0].corr()['target'].sort_values(ascending=False)
# correlation for males
male_corr = df[df['sex'] == 1].corr()['target'].sort_values(ascending=False)

print("\ntop correlations for females:")
print(female_corr.head(5))

print("\ntop correlations for males:")
print(male_corr.head(5))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import prepare_data

X_train, X_test, y_train, y_test, features = prepare_data('heart.csv')

# training the model with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# overall performance
y_pred = model.predict(X_test)
print("overall model performance")
print(classification_report(y_test, y_pred))

# gender-based performance analysis
test_indices = y_test.index
test_data_with_labels = pd.read_csv('heart.csv').loc[test_indices]

# separating predictions by gender
female_mask = (test_data_with_labels['sex'] == 0)
male_mask = (test_data_with_labels['sex'] == 1)

print("\nperformance for FEMALES")
print(classification_report(y_test[female_mask], y_pred[female_mask]))

print("\nperformance for MALES")
print(classification_report(y_test[male_mask], y_pred[male_mask]))

import shap

# creating a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
sex_col_idx = list(features).index('sex')
filtered_features = [f for f in features if f != 'sex']

shap_output = shap_values[1] if isinstance(shap_values, list) else shap_values

# visualizing for female patients
plt.close('all') 

female_shap = np.delete(shap_output[female_mask], sex_col_idx, axis=1)
female_data = np.delete(X_test[female_mask], sex_col_idx, axis=1)

shap.summary_plot(female_shap, female_data, feature_names=filtered_features, 
                  show=False, plot_size=(10, 6))

fig = plt.gcf() 
plt.title("feature importance for female patients", pad=10) 

# visualizing for male patients

male_shap = np.delete(shap_output[male_mask], sex_col_idx, axis=1)
male_data = np.delete(X_test[male_mask], sex_col_idx, axis=1)

shap.summary_plot(male_shap, male_data, feature_names=filtered_features, 
                  show=False, plot_size=(10, 6))

fig = plt.gcf()
plt.title("feature importance for male patients", pad=10)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

# mitigating bias with balanced class weights
fair_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
fair_model.fit(X_train, y_train)

# comparing confusion matrices for gender equality
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# female confusion matrix
ConfusionMatrixDisplay.from_estimator(
    model, X_test[female_mask], y_test[female_mask], 
    ax=axes[0], cmap='Blues', display_labels=['healthy', 'sick']
)
axes[0].set_title("confusion matrix for female patients")

# male confusion matrix
ConfusionMatrixDisplay.from_estimator(
    model, X_test[male_mask], y_test[male_mask], 
    ax=axes[1], cmap='Oranges', display_labels=['healthy', 'sick']
)
axes[1].set_title("confusion matrix for male patients")

plt.tight_layout()
plt.show()

# finding the "hidden" importance
y_fair_pred = fair_model.predict(X_test)
print("\nimproved performance for FEMALES (after bias mitigation)")
print(classification_report(y_test[female_mask], y_fair_pred[female_mask]))