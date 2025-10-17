import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from six import StringIO
from IPython.display import Image
import pydotplus

df = pd.read_csv('https://raw.githubusercontent.com/Nithin201342/All-datasets/refs/heads/main/Iris.csv')
print(df.head())

X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", min_samples_split=10)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print("Accuracy score: ", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True)
plt.show()
plot_tree(model, feature_names=X.columns)
plt.show()