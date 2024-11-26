import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('alzheimers_disease_data.csv')
df_subset=df[['FunctionalAssessment','ADL','MMSE', 'MemoryComplaints', 'BehavioralProblems','Diagnosis']]

# split the dataset
X = df_subset.drop('Diagnosis', axis=1)
y = df_subset['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred = rf.predict(X_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_resampled, y_train_resampled)
y_pred = gb.predict(X_test)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier(random_state=42)
ab.fit(X_train_resampled, y_train_resampled)
y_pred = ab.predict(X_test)

# random forest grid search cv
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

print(grid_search.best_params_)

# fitting the model with best parameters
rf_best = RandomForestClassifier(random_state=42, n_estimators=300, max_features='sqrt', max_depth=50, criterion='entropy')
rf_best.fit(X_train_resampled, y_train_resampled)
y_pred = rf_best.predict(X_test)

filename = 'random_forest_model.pkl'
pickle.dump(rf, open(filename, 'wb'))

filename = 'gradient_boosting_model.pkl'
pickle.dump(gb, open(filename, 'wb'))

filename = 'adaboost_model.pkl'
pickle.dump(ab, open(filename, 'wb'))

filename = 'random_forest_best_model.pkl'
pickle.dump(rf_best, open(filename, 'wb'))