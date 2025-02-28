import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("C:\\Users\\vandan_tank\\Documents\\Coding\\Movie-Revenue-Prediction\\data\\movies.csv")

df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
df['Lifetime Gross'] = pd.to_numeric(df['Lifetime Gross'], errors='coerce')

df = df.dropna()
df = df[(df['Budget'] > 0) & (df['Lifetime Gross'] > 0)]

scaler = StandardScaler()
df[['Budget', 'Lifetime Gross']] = scaler.fit_transform(df[['Budget', 'Lifetime Gross']])

X = df[['Budget']]
y = df['Lifetime Gross']

poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X, y)

y_pred = poly_model.predict(X)

r2 = r2_score(y, y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label="Actual Data")
plt.plot(np.sort(X, axis=0), np.sort(y_pred), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Budget (Normalized)")
plt.ylabel("Revenue (Normalized)")
plt.title("Box Office Revenue Prediction")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.text(min(X.values), max(y.values), f'R² Score: {r2:.3f}', fontsize=12, color='red')

plt.show()
