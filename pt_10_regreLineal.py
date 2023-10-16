import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
df = pd.read_csv('datos_procesados.csv')

# Eliminar las columnas DEATH_EVENT y age del dataframe para que sea la matriz X
X = df.drop(['DEATH_EVENT', 'age'], axis=1)

# Ajustar una regresión lineal sobre el resto de columnas y usar la columna age como vector y
y = df['age']

model = LinearRegression()
model.fit(X, y)

# Predice las edades y compara con las edades reales
y_pred = model.predict(X)

# Calcula el error cuadrático medio.
mse = mean_squared_error(y, y_pred)

print('Error cuadrático medio: ', mse)
