import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression

vueltas = pd.read_csv("../dataset_todos_pilotos/vueltas_spain_final.csv")
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

for iteracion in range (0, 20):

    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer="adam", loss= "mse") ##Cambiar esto para la siguiente reunion (positivo 1, negativo 0)
    model.summary()

    model.fit(trainX, trainY, epochs=12, batch_size=64, validation_split=0.0, verbose=1)


    y_real = testY.tolist()

    y_pred = model.predict(testX)  

    #Predecimos con los datos de entrenamiento, solo queremos sacar los puntos
    predictions = model.predict(trainX)


    clases = np.unique(trainY[:,0])

    puntos = []

    trainY_ = np.copy(trainY)
    index0 = trainY_[:,0] == clases[0]
    index1 = np.logical_or(trainY_[:,0] == clases[1],trainY_[:,0] == clases[2], trainY_[:,0] == clases[3])
    trainY_[index0] = 1
    trainY_[index1] = 0

    predictions = np.squeeze(predictions)
    predictions = predictions.reshape(-1,1)
    predictions.shape

    trainY_ = np.squeeze(trainY_)
    trainY_ = trainY_.reshape(-1,1)

    rl = LogisticRegression(random_state=0).fit(predictions, trainY_)
    punto = -rl.intercept_/rl.coef_
    puntos.append(punto)

    from sklearn.linear_model import LogisticRegression
    ##Este calcula el punto entre la clase 2 y la 3
    trainY_ = np.copy(trainY)
    index0 = np.logical_or(trainY_[:,0] == clases[0],trainY_[:,0] == clases[1] )
    index1 = np.logical_or(trainY_[:,0] == clases[2],trainY_[:,0] == clases[3] )
    trainY_[index0] = 0
    trainY_[index1] = 1

    predictions = np.squeeze(predictions)
    predictions = predictions.reshape(-1,1)
    predictions.shape

    trainY_ = np.squeeze(trainY_)
    trainY_ = trainY_.reshape(-1,1)

    rl = LogisticRegression(random_state=0).fit(predictions, trainY_)
    punto = -rl.intercept_/rl.coef_
    puntos.append(punto)

    trainY_ = np.copy(trainY)
    index0 = trainY_[:,0] != clases[3]
    index1 = trainY_[:,0] == clases[3]
    trainY_[index0] = 0
    trainY_[index1] = 1

    predictions = np.squeeze(predictions)
    predictions = predictions.reshape(-1,1)

    trainY_ = np.squeeze(trainY_)
    trainY_ = trainY_.reshape(-1,1)

    rl = LogisticRegression(random_state=0, penalty=None).fit(predictions, trainY_)
    punto = -rl.intercept_/rl.coef_
    puntos.append(punto)

    uniques = set(tuple(x) for x in y_real)
    uniques = [list(x) for x in uniques]


    #Esto es cogiendo el punto medio entre las dos clases con regresion logistica
    y_pred_normalized_rl = [] 
    for val in y_pred:
        if (val<puntos[0][0][0]):
            y_pred_normalized_rl.append(uniques[0][0])
        if (val>=puntos[0][0][0] and val<puntos[1][0][0]):
            y_pred_normalized_rl.append(uniques[3][0])
        if (val>=puntos[1][0][0] and val<puntos[2][0][0]):
            y_pred_normalized_rl.append(uniques[3][0])
        if (val>=puntos[2][0][0]):
            y_pred_normalized_rl.append(uniques[2][0])


    #Esto es cogiendo el punto medio
    y_pred_normalized_wrl = []
    for val in y_pred:
        dist = [abs(val-x[0]) for x in uniques]
        closest_val = uniques[np.argmin(dist)][0]
        y_pred_normalized_wrl.append(closest_val)


    bien_predecidas = 0
    mal_predecidas = 0

    for i in range (0, len(y_real)):
        if y_pred_normalized_rl[i] == y_real[i][0]:
            bien_predecidas += 1
        else:
            mal_predecidas += 1

    precision_rl = bien_predecidas/(mal_predecidas+bien_predecidas)      

    print("El modelo con rl tiene una precision de: ", precision_rl)

    bien_predecidas = 0
    mal_predecidas = 0

    for i in range (0, len(y_real)):
        if y_pred_normalized_wrl[i] == y_real[i][0]:
            bien_predecidas += 1
        else:
            mal_predecidas += 1

    precision_wrl = bien_predecidas/(mal_predecidas+bien_predecidas)   

    print("El modelo sin rl tiene una precision de: ", precision_wrl)

    clases = [str(sublista[0]) for sublista in uniques]

    y_real_ = [sublista[0] for sublista in y_real]


    y_real_str = [str(sublista) for sublista in y_real_]
    y_pred_normalized_str_wrl = [str(sublista) for sublista in y_pred_normalized_wrl]


    from sklearn.metrics import confusion_matrix, classification_report
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_real_str, y_pred_normalized_str_wrl, labels=clases)
    metricas=classification_report(y_real_str, y_pred_normalized_str_wrl, labels=clases)

    nombre_archivo = "ejecuciones/ejecucion_" + str(iteracion) + "_wrl.txt"

    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, "w") as archivo:

        # Redirigir la salida de print() al archivo
        print("Matriz de confusión:", file=archivo)
        print(clases, file=archivo)
        print(cm, file=archivo)
        print("Métricas:", file=archivo)
        print(metricas, file=archivo)
        print("Y real:", file=archivo)
        print(y_real_str, file = archivo)
        print("Y predicho:", file=archivo)
        print(y_pred_normalized_str_wrl, file = archivo)

    y_real_str = [str(sublista) for sublista in y_real_]
    y_pred_normalized_str_rl = [str(sublista) for sublista in y_pred_normalized_rl]


    cm = confusion_matrix(y_real_str, y_pred_normalized_str_rl, labels=clases)
    metricas_rl = classification_report(y_real_str, y_pred_normalized_str_rl, labels=clases)

    nombre_archivo = "ejecuciones/ejecucion_" + str(iteracion) + "_rl.txt"

    # Abrir el archivo en modo de escritura
    with open(nombre_archivo, "w") as archivo:

        # Redirigir la salida de print() al archivo
        print("Matriz de confusión:", file=archivo)
        print(clases, file=archivo)
        print(cm, file=archivo)

        print("Métricas:", file=archivo)
        print(metricas_rl, file=archivo)
        print("Y real:", file=archivo)
        print(y_real_str, file = archivo)
        print("Y predicho:", file=archivo)
        print(y_pred_normalized_str_rl, file = archivo)


