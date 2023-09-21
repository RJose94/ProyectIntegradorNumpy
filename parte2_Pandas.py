from datasets import load_dataset
import numpy as np
import pandas as pd

dataset = load_dataset("mstz/heart_failure")

data = dataset["train"]

edades = np.array(data['age'])

promedio_edad = np.mean(edades)

print(f"El promedio de edad de las personas participantes en el estudio es {promedio_edad}")

#Convertir en un DataFrame
df = pd.DataFrame(data)

#Separar el DataFrame en dos diferentes
df_dead = df[df['is_dead'] == 1]
df_alive = df[df['is_dead'] == 0]

#Calcular las edades de cada dataset e imprimir
promedio_edad_dead = df_dead['age'].mean()
promedio_edad_alive = df_alive['age'].mean()

print(f"El promedio de edad de las personas que perecieron en el estudio es {promedio_edad_dead}")
print(f"El promedio de edad de las personas que sobrevivieron en el estudio es {promedio_edad_alive}")