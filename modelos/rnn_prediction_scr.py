import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import statistics

pilotos = ['max_verstappen', 'hamilton', 'bottas']
resultados_totales = []
for piloto in pilotos:
	for j in range (1, 6):

		vueltas = pd.read_csv("../dataset_todos_pilotos/vueltas_spain_final.csv")
		vueltas.drop(['nextPit'], axis = 1, inplace = True)
		vueltas.drop(['makeStop'], axis = 1, inplace = True)
		vueltas_train = vueltas[vueltas['anyo'].isin([2018, 2019, 2020, 2021])]
		vueltas_test = vueltas[vueltas['anyo'] == 2022]
		vueltas_test = vueltas_test[vueltas_test['Piloto'] == piloto]
		vueltas_train = vueltas_train.drop(['anyo'], axis=1)
		vueltas_train.drop(['Piloto'], axis = 1, inplace = True)
		vueltas_test = vueltas_test.drop(['anyo'], axis=1)
		vueltas_test.drop(['Piloto'], axis = 1, inplace = True)
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
		    #print(vueltas_train_scaler[i + future - 1:i + future, 5])
		    
		for i in range (loopback, len(vueltas_test_scaler) -future +1):
		    testX.append(vueltas_test_scaler[i-loopback:i, 0:vueltas_test.shape[1]])
		    testY.append(vueltas_test_scaler[i + future - 1:i + future,5])
		    

		trainX, trainY, testX, testY = np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

		model = Sequential()
		model.add(SimpleRNN(trainX.shape[2],input_shape = (trainX.shape[1],trainX.shape[2]),return_sequences = True,
		                    activation = 'relu'))
		      
		model.add(SimpleRNN(256,activation = 'relu',return_sequences = True))
		model.add(Dropout(0.2))
		model.add(SimpleRNN(128,activation = 'relu', return_sequences= True))
		model.add(Dropout(0.2))
		model.add(SimpleRNN(64,activation = 'relu', return_sequences= False))
		model.add(Dropout(0.2))
		model.add(Dense(1))
		model.compile(optimizer ="adam", loss = 'mse')
		print(model.summary())

		history = model.fit(trainX,trainY,epochs = 20,batch_size = 64,validation_split =0.1,verbose = 1)
		plt.plot(history.history['loss'], label='Training loss')
		plt.plot(history.history['val_loss'], label='Validation loss')
		plt.legend()

		y_real = testY.tolist()
		y_pred = model.predict(testX)     

		uniques = set(tuple(x) for x in y_real)
		uniques = [list(x) for x in uniques]


		y_pred_normalized = []
		for val in y_pred:
		    dist = [abs(val-x[0]) for x in uniques]
		    closest_val = uniques[np.argmin(dist)][0]
		    y_pred_normalized.append(closest_val)

		clases = sorted([sublista[0] for sublista in uniques])
		y_real_ = [sublista[0] for sublista in y_real]
		y_real_str = [str(sublista) for sublista in y_real_]
		y_pred_normalized_str = [str(sublista) for sublista in y_pred_normalized]

		cr = classification_report(y_real_str, y_pred_normalized_str, labels=clases, output_dict=True)
		cr_2= classification_report(y_real_str, y_pred_normalized_str, labels=clases)

		nombre_archivo = f'pruebas_finales/pruebas_rnn_prediction/{piloto}/ejecucion_{j}.txt'
		with open(nombre_archivo, "w") as archivo:
		    print(cr_2, file=archivo)
		json_results = {
			"Piloto":piloto,
			"Iteration": str(j),
			"Accuracy": cr['micro avg']['precision'],
		}
		resultados_totales.append(json_results)


		with open("pruebas_finales/pruebas_rnn_prediction/global_metrics.json", 'w') as archivo_json:
			json.dump(resultados_totales, archivo_json)

accuracy_values = [result['Accuracy'] for result in resultados_totales]

# Calcula la media y la desviación estándar
mean_accuracy = statistics.mean(accuracy_values)
std_dev_accuracy = statistics.stdev(accuracy_values)

# Crea un diccionario para almacenar los resultados
global_results = {
"Media accuracy": mean_accuracy,
"Desviación estándar": std_dev_accuracy
}

# Guarda los resultados en un archivo de texto
with open('pruebas_finales/pruebas_rnn_prediction/metricas_finales.txt', 'w') as file:
	json.dump(global_results, file, indent=4)

print("Resultados guardados en metricas_finales.txt")









