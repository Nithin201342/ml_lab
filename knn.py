import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('https://raw.githubusercontent.com/THOMASBAIJU/Dataset/refs/heads/main/social-network-ads.csv')
print(df.head())

print(df.isnull().sum())
X = df.drop(['User ID','Purchased'], axis=1)
y = df['Purchased']

X = pd.get_dummies(X)
print(X)

model = StandardScaler()
scaled = model.fit_transform(X)
X_scaled = pd.DataFrame(scaled)
X_scaled.columns = X.columns
print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Accuracy score(Train): ", accuracy_score(y_train, y_train_pred), "\nAccuracy score(Test): ", accuracy_score(y_test, y_test_pred))
print("Classification Report(Train): \n", classification_report(y_train, y_train_pred), "\nClassification Report(Test): \n", classification_report(y_test, y_test_pred))
