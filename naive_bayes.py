import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/Nithin201342/All-datasets/refs/heads/main/Iris.csv')

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score: ", accuracy_score(y_pred, y_test))
print("Confusion matrix: \n", confusion_matrix(y_pred, y_test))