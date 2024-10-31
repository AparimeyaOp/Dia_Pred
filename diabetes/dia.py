import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sb
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('diabetes.csv')
data = data.drop_duplicates()
data[['Glucose', 'Blood_Pressure', 'Age', 'BMI', 'Skin_Thickness', 'Insulin', 'Pregnancies']] = \
    data[['Glucose', 'Blood_Pressure', 'Age', 'BMI', 'Skin_Thickness', 'Insulin', 'Pregnancies']].replace(0, pd.NA)
data.fillna(data.median(), inplace=True)

# Display data information
print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.describe())

# Split the data
X = data[['Glucose', 'Blood_Pressure', 'Age', 'BMI', 'Skin_Thickness', 'Insulin', 'Pregnancies']]
Y = data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=40)

# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test)
print("Random forest classification report: ")
print(classification_report(Y_test, y_pred_rf))

print("Confusion matrix: ")
cm = confusion_matrix(Y_test, y_pred_rf)
print(cm)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, Y_train)
print(f"Best parameters: {grid_search.best_params_}")

best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, Y_train)

# Make predictions with optimized model
y_pred_opti = best_rf_model.predict(X_test)
print("Optimized Random forest classification report: ")
print(classification_report(Y_test, y_pred_opti))

print("Optimized Confusion matrix: ")
optimized = confusion_matrix(Y_test, y_pred_opti)
print(optimized)

# Visualize glucose distribution
sb.histplot(data['Glucose'], kde=True)
plt.title('GLUCOSE DISTRIBUTION')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.show()

# Save the model
joblib.dump(best_rf_model, 'diabetes_model.pkl')
print("Model saved as 'diabetes_model.pkl'")
