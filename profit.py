import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("data.csv")
print(data.head())
print(data.describe())
x = data[["R&D Spend", "Administration", "Marketing Spend"]]
y = data["Profit"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Profit": ypred.flatten()})
print(data.head())