import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# colunas corretas do dataset
X = lifesat[["GDP per capita"]].values
Y = lifesat[["Life satisfaction"]].values

# gráfico
lifesat.plot(
    kind="scatter",
    grid=True,
    x="GDP per capita",
    y="Life satisfaction"
)
plt.show()

# modelo
model = LinearRegression()
model.fit(X, Y)

# previsão
X_new = [[37655.2]]
print(model.predict(X_new))