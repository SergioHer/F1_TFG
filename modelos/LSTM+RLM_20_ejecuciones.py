import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json


vueltas = pd.read_csv("../dataset_todos_pilotos/vueltas_saudi_arabia_final.csv")
vueltas.drop(['Piloto'], axis = 1, inplace = True)
vueltas_train = vueltas[vueltas['anyo'].isin([2018, 2019, 2020, 2021])]
vueltas_test = vueltas[vueltas['anyo'] == 2022]
vueltas_train = vueltas_train.drop(['anyo'], axis=1)
vueltas_test = vueltas_test.drop(['anyo'], axis=1)

vueltas_train = vueltas_train.astype(float)
vueltas_test = vueltas_test.astype(float)

scaler = MinMaxScaler()
scaler.fit(vueltas_train)
vueltas_train_scaler = scaler.transform(vueltas_train)
vueltas_test_scaler = scaler.transform(vueltas_test)

trainX = []
trainY = []
testX = []
testY = []
loopback = 10 # Esto es el numero de muestras que usara en el pasado 
future = 1 # Esto es el numero de hechos futuros que usará como salida a las 10 muestras del pasado

for i in range (loopback, len(vueltas_train_scaler) -future +1):
    trainX.append(vueltas_train_scaler[i-loopback:i, 0:vueltas_train.shape[1]])
    trainY.append(vueltas_train_scaler[i + future - 1:i + future, 5])
    
for i in range (loopback, len(vueltas_test_scaler) -future +1):
    testX.append(vueltas_test_scaler[i-loopback:i, 0:vueltas_test.shape[1]])
    testY.append(vueltas_test_scaler[i + future - 1:i + future,5])
trainX, trainY, testX, testY = np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

resultados_totales = []

for iteracion in range (0, 20):
    print("******************************* ITERATION " + str(iteracion) + " *******************************")
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer="adam", loss= "mse")
    model.summary()
    history = model.fit(trainX, trainY, epochs=12, batch_size=64, validation_split=0.0, verbose=1)

    clases = np.unique(trainY[:,0])

    f = lambda x: np.argwhere(x == clases)
    f = np.vectorize(f)
    y_real = f(testY[:,0])

    predictions = model.predict(testX)
    plt.scatter(predictions[:,0], predictions[:,0], s=1, c=y_real+1)

    X_train, X_test, y_train, y_test = train_test_split(predictions, y_real, test_size=0.3, stratify=y_real)

    rl = LogisticRegression(random_state=0, penalty=None).fit(X_train, y_train)
    pred_test_2 = rl.predict(X_test)

    cm = confusion_matrix(y_test, pred_test_2)
    metricas = classification_report(y_test, pred_test_2)
    metricas2 = classification_report(y_test, pred_test_2, output_dict=True)


    nombre_archivo = "ejecuciones/ejecuciones_saudi_arabia/ejecucion_" + str(iteracion) + "_lstm+rlm.txt"

    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, "w") as archivo:
        print("Matriz de confusión:", file=archivo)
        print(clases, file=archivo)
        print(cm, file=archivo)
        print("Métricas:", file=archivo)
        print(metricas, file=archivo)

    json_results = {
        "Iteration": str(iteracion),
        "Accuracy": metricas2['accuracy'],
        "Macro_avg": metricas2['macro avg'],
        "Weighted_avg": metricas2['weighted avg']
    }
    resultados_totales.append(json_results)
    with open("ejecuciones/ejecuciones_saudi_arabia/global_json_saudi_arabia.json", 'w') as archivo:
        json.dump(resultados_totales, archivo)


