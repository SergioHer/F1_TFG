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

		[**************CODIGO DEL MODELO ESPECIFICO**************]

		nombre_archivo = f'pruebas_finales/pruebas_gru_prediction/{piloto}/ejecucion_{j}.txt'
		with open(nombre_archivo, "w") as archivo:
		    print(cr_2, file=archivo)
		json_results = {
			"Piloto":piloto,
			"Iteration": str(j),
			"Accuracy": cr['micro avg']['precision'],
		}
		resultados_totales.append(json_results)


		with open("pruebas_finales/pruebas_gru_prediction/global_metrics.json", 'w') as archivo_json:
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
with open('pruebas_finales/pruebas_gru_prediction/metricas_finales.txt', 'w') as file:
	json.dump(global_results, file, indent=4)

print("Resultados guardados en metricas_finales.txt")









