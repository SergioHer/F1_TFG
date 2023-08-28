import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, GRU
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

		vueltas = pd.read_csv("../dataset_todos_pilotos/final/laps_spain_final_v2.csv")
		vueltas.drop(['nextPit'], axis = 1, inplace = True)
		vueltas['makeStop'] = vueltas['makeStop'].shift(1)   
		vueltas['makeStop'][0] = 0
		vueltas_train = vueltas[vueltas['anyo'].isin([2018, 2019, 2020])]
		vueltas_validation = vueltas[vueltas['anyo'].isin([2021])]
		vueltas_test = vueltas[vueltas['anyo'] == 2022]
		vueltas_test = vueltas_test[vueltas_test['Piloto'] == piloto]
		vueltas_validation = vueltas_validation.drop(['anyo'], axis=1)
		vueltas_test = vueltas_test.drop(['anyo'], axis=1)
		vueltas_train.drop(['Piloto'], axis = 1, inplace = True)
		vueltas_test.drop(['Piloto'], axis = 1, inplace = True)
		vueltas_validation.drop(['Piloto'], axis = 1, inplace = True)
		vueltas_train = vueltas_train.drop(['Stint'], axis=1)
		vueltas_validation = vueltas_validation.drop(['Stint'], axis=1)
		vueltas_test = vueltas_test.drop(['Stint'], axis=1)
		vueltas_train = vueltas_train.astype('float')
		vueltas_validation = vueltas_validation.astype(float)
		vueltas_test = vueltas_test.astype(float)


		vueltas_train_18 = vueltas_train[vueltas_train['anyo'] == 2018]
		vueltas_train_18 = vueltas_train_18.drop('anyo', axis=1)
		vueltas_train_18['makeStop'] = vueltas_train_18['makeStop'].astype('int32')
		scaler = MinMaxScaler()
		scaler.fit(vueltas_train_18)
		vueltas_train_scaler_18= scaler.transform(vueltas_train_18)


		vueltas_train_19 = vueltas_train[vueltas_train['anyo'] == 2019]

		vueltas_train_19 = vueltas_train_19.drop('anyo', axis=1)
		vueltas_train_19['makeStop'] = vueltas_train_19['makeStop'].astype('int32')

		scaler.fit(vueltas_train_19)
		vueltas_train_scaler_19= scaler.transform(vueltas_train_19)


		vueltas_train_20 = vueltas_train[vueltas_train['anyo'] == 2020]

		vueltas_train_20 = vueltas_train_20.drop('anyo', axis=1)
		vueltas_train_20['makeStop'] = vueltas_train_20['makeStop'].astype('int32')

		scaler.fit(vueltas_train_20)
		vueltas_train_scaler_20= scaler.transform(vueltas_train_20)

		vueltas_test_scaler = scaler.transform(vueltas_test)
		vueltas_validation_scaler = scaler.transform(vueltas_validation)

		trainX = []
		trainY = []
		testX = []
		testY = []
		validationX = []
		validationY = []
		loopback = 8 # Esto es el numero de muestras que usara en el pasado 
		future = 1 # Esto es el numero de hechos futuros que usará como salida a las 10 muestras del pasado

		for i in range (loopback, len(vueltas_train_scaler_18) -future +1):
		    trainX.append(vueltas_train_scaler_18[i-loopback:i, 0:vueltas_train_18.shape[1]])
		    trainY.append(vueltas_train_scaler_18[i + future - 1:i + future,5])
		    
		for i in range (loopback, len(vueltas_train_scaler_19) -future +1):
		    trainX.append(vueltas_train_scaler_19[i-loopback:i, 0:vueltas_train_19.shape[1]])
		    trainY.append(vueltas_train_scaler_19[i + future - 1:i + future,5])
		    
		for i in range (loopback, len(vueltas_train_scaler_20) -future +1):
		    trainX.append(vueltas_train_scaler_20[i-loopback:i, 0:vueltas_train_20.shape[1]])
		    trainY.append(vueltas_train_scaler_20[i + future - 1:i + future,5])
		 
		for i in range (loopback, len(vueltas_test_scaler) -future +1):
		        testX.append(vueltas_test_scaler[i-loopback:i, 0:vueltas_test.shape[1]])
		        testY.append(vueltas_test_scaler[i + future - 1:i + future,5])
		        
		    
		for i in range (loopback, len(vueltas_validation_scaler) -future +1):
		    validationX.append(vueltas_validation_scaler[i-loopback:i, 0:vueltas_validation.shape[1]])
		    validationY.append(vueltas_validation_scaler[i + future - 1:i + future,5])


		trainX, trainY, testX, testY, validationX, validationY = np.array(trainX), np.array(trainY), np.array(testX), np.array(testY), np.array(validationX), np.array(validationY)

		trainX.shape[0]


		testX.shape

		model = Sequential()
		model.add(LSTM(256, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(128, activation='tanh', return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(optimizer="adam", loss= "binary_crossentropy")
		model.summary()


		history = model.fit(trainX, trainY, epochs=15, batch_size=64, validation_split=0.2, verbose=1)

		plt.plot(history.history['loss'], label='Training loss')
		plt.legend()


		predictions_test = model.predict(testX)

		clases = np.unique(validationY[:,0]) 

		f = lambda x: np.argwhere(x == clases)
		f = np.vectorize(f)
		y_real = f(validationY[:,0])


		predictions = model.predict(validationX)


		plt.scatter(predictions[:,0], predictions[:,0], s=1, c=y_real+1)

		y_real_train = f(trainY[:,0])

		y_real_train

		predictions_test = model.predict(testX)

		y_real_test =  f(testY[:,0])

		from sklearn.linear_model import LogisticRegression
		rl = LogisticRegression(random_state=0, penalty=None).fit(predictions, y_real) # Esto es de validacion

		pred_test_2 = rl.predict(predictions_test)

		cr = classification_report(y_real_test, pred_test_2, output_dict=True)
		cr_2= classification_report(y_real_test, pred_test_2)

		nombre_archivo = f'pruebas_finales/pruebas_lstm_rl_bcr/{piloto}/ejecucion_{j}.txt'
		with open(nombre_archivo, "w") as archivo:
		    print(cr_2, file=archivo)
		json_results = {
			"Piloto":piloto,
			"Iteration": str(j),
			"Accuracy": cr['accuracy'],
		}
		resultados_totales.append(json_results)


		with open("pruebas_finales/pruebas_lstm_rl_bcr/global_metrics.json", 'w') as archivo_json:
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
with open('pruebas_finales/pruebas_lstm_rl_bcr/metricas_finales.txt', 'w') as file:
	json.dump(global_results, file, indent=4)

print("Resultados guardados en metricas_finales.txt")









