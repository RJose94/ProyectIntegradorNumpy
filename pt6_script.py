import pandas as pd
import requests
import numpy as np
import sys

def descargar_csv(url):
    respuesta = requests.get('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv')
    nombre_archivo = url.split("/")[-1]  # Extrae el nombre del archivo de la URL

    with open(nombre_archivo, 'w') as archivo:
        archivo.write(respuesta.text)

    return pd.read_csv(nombre_archivo)

def categorizar_datos(df):
    #Categorizar por edad
    bins = [0, 12, 19, 39, 59, np.inf]
    labels = ['Niño', 'Adolescente', 'Jóvenes adulto', 'Adulto', 'Adulto mayor']
    df['categoria_edad'] = pd.cut(df['age'], bins=bins, labels=labels)

    return df

def main():
    if len(sys.argv) < 2:
        print("Por favor, proporciona una URL como argumento al ejecutar este script.")
        return

    url = sys.argv[1]
    
    df = descargar_csv(url)
    
    df_categorizado = categorizar_datos(df)

    # Guardar el resultado como csv
    df_categorizado.to_csv('datos_procesados_2.csv', index=False)

if __name__ == '__main__':
    main()
#Para acceder al script por terminal
# python pt6_script.py https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv
