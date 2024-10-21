import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import resample
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score
import shap

# Reading the data
data = pd.read_csv('thyroid_data.csv')

# Print the first five rows of the data
print(data.head())

# Shape of the data
print("Data Shape:", data.shape)

# Check the counts for each category
categories = ['hyperthyroid', 'hypothyroid', 'sick', 'negative']
for category in categories:
    print(f"No of {category} in Dataset:", len(data[data['Category'] == category]))

# Columns with '?' values
for column in data.columns:
    count = data[column][data[column] == '?'].count()
    if count != 0:
        print(f"{column}: {count}")

# Drop unnecessary columns
data = data.drop(['S.no', 'On Thyroxine', 'Query on Thyroxine', 'On Antithyroid Medication', 'I131 Treatment', 
                  'Query Hypothyroid', 'Query Hyperthyroid', 'Lithium', 'TSH Measured', 'Hypopituitary', 'Psych', 
                  'T3 Measured', 'TT4 Measured', 'T4U Measured', 'FTI Measured'], axis=1)

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert columns to numeric, coercing errors to NaN
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['TSH'] = pd.to_numeric(data['TSH'], errors='coerce')
data['T3'] = pd.to_numeric(data['T3'], errors='coerce')
data['TT4'] = pd.to_numeric(data['TT4'], errors='coerce')
data['T4U'] = pd.to_numeric(data['T4U'], errors='coerce')
data['FTI'] = pd.to_numeric(data['FTI'], errors='coerce')

# Fill NaN values with median
data['Age'] = data['Age'].fillna(data['Age'].median())
data['TSH'] = data['TSH'].fillna(data['TSH'].median())
data['T3'] = data['T3'].fillna(data['T3'].median())
data['TT4'] = data['TT4'].fillna(data['TT4'].median())
data['T4U'] = data['T4U'].fillna(data['T4U'].median())
data['FTI'] = data['FTI'].fillna(data['FTI'].median())

# One-hot encode categorical variables
categorical_vars = ['Sex', 'Sick', 'Pregnant', 'Thyroid Surgery', 'Goitre', 'Tumor']
data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# Drop 'TSH' due to skewness
data = data.drop(['TSH'], axis=1)

# Encode target variable
lblEn = LabelEncoder()
data['Category'] = lblEn.fit_transform(data['Category'])

# Check for NaN values
print("Missing values after preprocessing:\n", data.isnull().sum())

# Split data into features and target
X = data.drop(['Category'], axis=1)
y = data['Category']

# Apply SMOTEENN for handling imbalanced data
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# RandomForest Classifier
classifier_forest = RandomForestClassifier(criterion='entropy', random_state=0)
classifier_forest.fit(X_train, y_train)

# Model performance
y_pred = classifier_forest.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Cross-Validation Score:\n", cross_val_score(classifier_forest, X_train, y_train, cv=10).mean())

# Save the model
filename = 'thyroid_model.pkl'
pickle.dump(classifier_forest, open(filename, 'wb'))

# Feature names for the prediction data
feature_names = ['Age', 'T3', 'TT4', 'T4U', 'FTI', 'Sex_M', 'Sick_t', 'Pregnant_t', 'Thyroid Surgery_t', 'Goitre_t', 'Tumor_t']

# Testing the model with new data using DataFrame with feature names
new_data = pd.DataFrame([[41, 2.5, 125, 1.14, 109, 0, 0, 0, 0, 0, 0]], columns=feature_names)
print(classifier_forest.predict(new_data))

new_data = pd.DataFrame([[63, 5.5, 199, 1.05, 190, 0, 0, 0, 0, 0, 0]], columns=feature_names)
print(classifier_forest.predict(new_data))

new_data = pd.DataFrame([[44, 1.4, 39, 1.16, 33, 1, 0, 0, 0, 0, 0]], columns=feature_names)
print(classifier_forest.predict(new_data))

new_data = pd.DataFrame([[61, 1, 96, 0.93, 109, 1, 1, 0, 0, 0, 0]], columns=feature_names)
print(classifier_forest.predict(new_data))

# SHAP for model interpretability
explainer = shap.TreeExplainer(classifier_forest)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)
