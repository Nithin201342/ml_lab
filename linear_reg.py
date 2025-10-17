import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://raw.githubusercontent.com/Nithin201342/Advertising-dataset/refs/heads/main/advertising.csv')
print(df.head())
print(df.isnull().sum())
plt.title('Linear Regression')
plt.xlabel('TV')
plt.ylabel('Sales')
sns.regplot(x=df['TV'], y=df['Sales'], data=df, line_kws={'color':'red'})
plt.show()

X = df[['TV']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
c = model.intercept_
print('Slope: ', m, '\nIntercept: ', c)

y_pred = model.predict(X_test)
res = r2_score(y_test, y_pred)
print('R2 Score: ', res)
mse = mean_squared_error(y_test, y_pred)
print('MSE: ', mse)
mae = mean_absolute_error(y_test, y_pred)
print('MAE', mae)