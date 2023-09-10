# Importar las bibliotecas necesarias
from datasets import load_dataset
import numpy as np

# Descargar el conjunto de datos
dataset = load_dataset("mstz/heart_failure")

# Acceder a la partición de entrenamiento
data = dataset["train"]

# Convertir la lista de edades a un arreglo de NumPy
edades = np.array(data['age'])

# Calcular el promedio de edad
promedio_edad = np.mean(edades)

print(f"El promedio de edad de las personas participantes en el estudio es {promedio_edad}")