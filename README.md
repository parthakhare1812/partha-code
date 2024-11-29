# partha-code

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('winequality-red.csv', sep=';')


print("Columns in the dataset:")
print(data.columns)

print("\nCorrelation matrix:")
print(data.corr().to_string())


X = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides"]]
Y_regression = data["quality"]


sns.pairplot(data, vars=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "quality"])
plt.show()


X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X, Y_regression, test_size=0.30, random_state=1)


print("\nPerforming Linear Regression...")
reg_model = LinearRegression()
reg_model.fit(X_train_reg, Y_train_reg)
Y_pred_reg = reg_model.predict(X_test_reg)


mse = mean_squared_error(Y_test_reg, Y_pred_reg)
r2 = r2_score(Y_test_reg, Y_pred_reg)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")




data['quality_class'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)  # 1 for high quality, 0 for low quality


Y_classification = data['quality_class']


X_train_clf, X_test_clf, Y_train_clf, Y_test_clf = train_test_split(X, Y_classification, test_size=0.30, random_state=1)


print("\nPerforming Logistic Regression...")
log_model = LogisticRegression()
log_model.fit(X_train_clf, Y_train_clf)
Y_pred_log = log_model.predict(X_test_clf)


print("\nPerforming KNN Analysis...")
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can tune `n_neighbors`
knn_model.fit(X_train_clf, Y_train_clf)
Y_pred_knn = knn_model.predict(X_test_clf)


def evaluate_model(model_name, Y_true, Y_pred):
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(Y_true, Y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_true, Y_pred))
    print("Classification Report:")
    print(classification_report(Y_true, Y_pred))


evaluate_model("Logistic Regression", Y_test_clf, Y_pred_log)

evaluate_model("KNN", Y_test_clf, Y_pred_knn)
coefficients = pd.DataFrame(log_model.coef_[0], X.columns, columns=['Coefficient'])
coefficients.plot(kind='bar', title='Logistic Regression Coefficients')
plt.show()
