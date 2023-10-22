import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('datos_procesados.csv')

# Verificar si la columna 'smoking' existe en el DataFrame
if 'smoking' in df.columns:
    # Eliminar la columna categoria_edad del dataframe para que sea la matriz X
    X = df.drop(['categoria_edad'], axis=1)

    # Ajustar una regresión lineal sobre el resto de columnas y usar la columna smoking como vector y
    y = df['smoking']

    # Graficar la distribución de clases
    plt.hist(y)
    plt.show()

    # Realizar la partición del dataset en conjunto de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ajustar un random forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predice las clases y compara con las clases reales
    y_pred = model.predict(X_test)

    # Calcula el accuracy.
    acc = accuracy_score(y_test, y_pred)
    
    # Calcula F1-Score.
    f1 = f1_score(y_test, y_pred)

    # Calcula la matriz de confusión.
    cm = confusion_matrix(y_test, y_pred)

    print('Accuracy: ', acc)
    print('F1-Score: ', f1)
    print('Confusion Matrix: \n', cm)
else:
    print("La columna 'smoking' no existe en el DataFrame. Por favor verifica tus datos.")
