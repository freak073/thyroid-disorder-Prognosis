import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from warnings import filterwarnings
filterwarnings(action='ignore')

# Load and preprocess the data
thyroid_data = pd.read_csv("thyroid_data.csv")
thyroid_data = thyroid_data.drop(['S.no'], axis=1)

# Fix incorrect age entry
thyroid_data.loc[thyroid_data['Age'] == '455', 'Age'] = '45'

# Drop unnecessary columns
thyroid_data = thyroid_data.drop(['TSH Measured', 'T3 Measured', 'TT4 Measured', 'T4U Measured', 'FTI Measured'], axis=1)

# Binarize Category Columns
def convert_category(dataframe, column):
    if column == 'Sex':
        dataframe.loc[dataframe[column] == 'F', column] = 0
        dataframe.loc[dataframe[column] == 'M', column] = 1
    else:
        dataframe.loc[dataframe[column] == 'f', column] = 0
        dataframe.loc[dataframe[column] == 't', column] = 1

binary_cols = ['Age', 'Sex', 'On Thyroxine', 'Query on Thyroxine',
               'On Antithyroid Medication', 'Sick', 'Pregnant', 'Thyroid Surgery',
               'I131 Treatment', 'Query Hypothyroid', 'Query Hyperthyroid', 'Lithium',
               'Goitre', 'Tumor', 'Hypopituitary', 'Psych', 'TSH', 'T3', 'TT4', 'T4U',
               'FTI']

for col in binary_cols:
    convert_category(thyroid_data, col)

# Convert '?' to np.nan and convert numeric data to numeric dtype
for col in thyroid_data.columns:
    if col != 'Category':
        thyroid_data.loc[thyroid_data[col] == '?', col] = np.nan
        thyroid_data[col] = pd.to_numeric(thyroid_data[col])

# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data = imputer.fit_transform(thyroid_data.drop('Category', axis=1))
imputed_data = pd.DataFrame(imputed_data, columns=thyroid_data.columns.drop('Category'))

thyroid_data = pd.concat([imputed_data, thyroid_data['Category']], axis=1)

# Split the data
X = thyroid_data.drop('Category', axis=1)
y = thyroid_data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Normalize the data
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Using RandomForest with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate the model
predictions = best_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
pickle.dump(best_model, open('thyroid_model_rf.pkl', 'wb'))

# Load the model to compare the results
loaded_model = pickle.load(open('thyroid_model_rf.pkl', 'rb'))
print(loaded_model.predict([[20, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 2.5, 200, 2, 200]]))
