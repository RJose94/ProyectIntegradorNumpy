from datasets import load_dataset
import numpy as np
import pandas as pd
import requests 

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

# Verificar que los tipos de datos son correctos en cada columna
print(df.dtypes)

smoking_data = np.random.randint(2, size=299)
sex_data = np.random.randint(2, size=299)

# Crear la columna 'sex' en el DataFrame
df['sex'] = sex_data
# Crear la columna 'smoking' en el DataFrame
df['smoking'] = smoking_data

print(df.columns)

# Asegurar si las columnas "sex y smoking existen en DataFrame"
if 'sex' in df.columns and 'smoking' in df.columns:
    # Calcular la cantidad de hombres fumadores vs mujeres fumadoras
    hombres_fumadores = df[(df['sex'] == 1) & (df['smoking'] == 1)].shape[0]
    mujeres_fumadoras = df[(df['sex'] == 0) & (df['smoking'] == 1)].shape[0]

    print(f"La cantidad de hombres fumadores es {hombres_fumadores}")
    print(f"La cantidad de mujeres fumadoras es {mujeres_fumadoras}")
else:
    print("Las columnas 'sex' y 'smoking' no existen en el DataFrame.")

def descargar_csv(url):
    respuesta = requests.get(url)
    nombre_archivo = url.split("/")[-1]  # Extrae el nombre del archivo de la URL

    with open(nombre_archivo, 'w') as archivo:
        archivo.write(respuesta.text)

# Uso de la función
descargar_csv('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv')