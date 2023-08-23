
import fastf1
import pandas as pd
import requests
import json
import time
#Obtención de todos los circuitos en csv

circuitos = pd.read_csv("../circuits/circuits_2022.csv")

#Limpiamos el dataframe de circuitos
print("Circuitos disponibles: ")
circuitos = circuitos.drop(['url', 'Location/lat', 'Location/long', 'Location/locality'], axis = 1)
print(circuitos['Location/country'].to_string(index=False))


#Creamos un nuevo dataframe con el nombre, la localidad y el nivel de desgaste que pueden tener los neumaticos
circuitos_desgaste = circuitos.drop(['circuitId'], axis=1)

#Creamos una lista con el nivel de desgaste, fuente PIRELLI F1 2022
desgaste = ['Bajo', 'Alto', 'Alto', 'Bajo', 'Alto', 'Bajo', 'Medio', 'Medio', 'Bajo', 'Bajo', 'Bajo', 'Medio', 'Medio', 'Medio', 'Medio', 'Bajo', 'Medio', 'Alto', 'Medio', 'Bajo', 'Medio', 'Medio']
circuitos_desgaste['desgaste'] = desgaste

vueltas_circuito = ['58', '56', '57', '51', '66', '70', '63', '71', '50', '61', '57', '78', '53', '71', '53', '71', '52', '44', '53', '70', '58', '72']
circuitos_desgaste['vueltas_totales'] = vueltas_circuito

km_circuito = ['307.574', '308.405', '308.238', '306.049', '307.104', '306.630', '309.049', '305.879', '308.450', '308.706', '308.326', '260.286', '306.720', '306.452', '309.690', '305.354', '306.198', '308.052', '307.471', '305.270', '305.355', '306.648']
circuitos_desgaste['km_total'] = km_circuito

#Vamos a preguntar por el piloto que queramos, el circuito que queramos y el año que queramos:

#Print de circuitos disponibles
nombreCircuito = input("Introduce el pais de un Gran Premio: ")

anyoGp = input("Introduce el año del GP (2017-2022): ")


#Vamos entonces a crear un dataframe para un piloto en una determinada 
#carrera, por ejemplo, Fernando Alonso y Monaco 2022
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #Omitimos los future warning

fastf1.Cache.enable_cache('../cache') # Para cachear los datos y no tener que bajarlos todo el tiempo

#A continuacion, nos guardamos la sesión en una variable de tipo sesion

race = fastf1.get_session(int(anyoGp), nombreCircuito, 'R')
race.load(weather=True, telemetry= True)

#Print de pilotos disponibles
print("Pilotos disponibles: ")
race.results['Abbreviation']
print(race.results['Abbreviation'].to_string(index=False))

nombrePiloto = input("Introduce un piloto: ")

#Creamos un diccionario para tener los nombres de los pilotos en Ergast y en FastF1 (para poder sacar la posicion por cada vuelta)
#ya que la api tiene la pinta del valor y fastf1 tiene la pinta de la clave (en el diccionario)
pilotosDic = {"HAM":"hamilton", "VER":"max_verstappen", "BOT": "bottas", "NOR":"norris", "PER":"perez", "LEC":"leclerc",
             "RIC":"ricciardo", "SAI":"sainz", "TSU":"tsunoda", "STR":"stroll", "RAI":"raikkonen", "GIO":"giovinazi",
              "OCO":"ocon", "RUS":"russel", "VET":"vettel", "MSC":"mick_schumacher", "GAS":"GASLY", "LAT":"latifi",
             "ALO":"alonso", "MAZ":"mazepin"}

laps_race_pilot = race.laps.pick_driver(nombrePiloto)
weather = race.weather_data # Cogemos tambien el tiempo que hizo al inicio de la carrera

#Limpiamos el dataset con solo lo que nos interesa

laps_race_pilot = laps_race_pilot.filter(['Time', 'LapTime', 'LapNumber', 'Compound', 'TyreLife', 'FreshTyre', 'TrackStatus', 'Stint', 'PitInTime'])


#De aqui solo queremos la temperatura de la pista, la temperatura ambiente y si llovía o no 
#Importante, hacemos match de la columna time, o ponemos simplemente el tiempo que hacia en la salida????
weather = weather.filter(['Time', 'AirTemp', 'Rainfall', 'TrackTemp'])

weather['Time'] = weather['Time'].astype(str)

#Modificamos la columna de Time, para que solo nos muestre la hora y el minuto, esto es para
#que casen las keys al hacer el join y podamos tener info de la temperatura minuto a minuto en las vueltas
weather['Time'] = weather["Time"].apply(lambda x: x[7:12])


#Hacemos lo mismo para la columa Time de las vueltas
laps_race_pilot['Time'] = laps_race_pilot['Time'].astype(str)
laps_race_pilot['Time'] = laps_race_pilot["Time"].apply(lambda x: x[7:12])

#Hacemos el join para meter en el mismo dataset, la imformacion en cada vuelta de la temperatura ambiente y de la pista
laps_race_pilot = laps_race_pilot.set_index('Time').join(weather.set_index('Time'))

laps_race_pilot['LapTime'] = laps_race_pilot['LapTime'].astype(str)
laps_race_pilot["LapTime"] = laps_race_pilot["LapTime"].apply(lambda x: x[10:19] if x!="NaT" else '0')


#Para la posicion de salida
posicion_salida = race.get_driver(nombrePiloto)['GridPosition']


#Necesitamos ahora los outputs, que es la siguiente vuelta a parar
vueltas_parada = laps_race_pilot[laps_race_pilot['PitInTime'].notna()]


nVuelta_parada = vueltas_parada['LapNumber']
nParadas = 0
paradas = []
for i in nVuelta_parada:
    paradas.append(i)
    nParadas = nParadas + 1


#Eliminanos la variable de la hora a la que entra a box porque ya no es necesaria
laps_race_pilot.drop(['PitInTime'], axis = 1, inplace=True)


#Ahora hay que añadir al dataframe las variables
# - Posicion de salida
# - Siguiente parada (dependera de la vuelta en la que se encuentre)
# - Todas variables del circuito en el que nos encontramos, en este caso Barcelona
# - No creo que haga falta el piloto, las estrategias son independientes del piloto (o eso voy a pensar... que no será asi)

#Aqui ahora tenemos que hacer un match de el nombre que ha introducido el usuario, con el nombre que salga en el dataset
circuito_actual = circuitos_desgaste.loc[circuitos_desgaste['Location/country'].str.lower() == nombreCircuito.lower()]
circuito_actual = circuito_actual.filter(['circuitName', 'desgaste', 'vueltas_totales', 'km_total'])



roundNumber = race.event.RoundNumber
pilotoNombre=pilotosDic[nombrePiloto]
posicionesList = []


#El numero de vuelta es una iteracion a cada fila del dataframe de vueltas del piloto

for index, row in laps_race_pilot.iterrows():
    nVuelta = int(row['LapNumber'])
    #Construimos la URL de la api
    api = "http://ergast.com/api/f1/" + anyoGp + "/" + str(roundNumber) + "/drivers/" +pilotoNombre+ "/laps/" + str(nVuelta) +".json"
    response = requests.get(api)
    response_dict = json.loads(response.text)
    posicion = response_dict['MRData']['RaceTable']['Races'][0]['Laps'][0]['Timings'][0]['position']
    posicionesList.append(posicion)
    time.sleep(0.5)

laps_race_pilot['posicionActual'] = posicionesList



#Hay que hacer una funcion para sacar el numero de vueltas (nVueltas) que se ha dado en ese circuito, y repetir nVueltas para crear una lista con el valor
nVueltasCircuito=laps_race_pilot.tail(1).LapNumber.values[0]
circuitName = circuito_actual.circuitName.values[0]
desgaste = circuito_actual.desgaste.values[0]
km_total = circuito_actual.km_total.values[0]
vueltas_totales = circuito_actual.vueltas_totales.values[0]
desgaste

    
#La salida se la tenemos que meter a todas las filas del dataframe
laps_race_pilot['posicionSalida'] = posicion_salida
laps_race_pilot['circuitName'] = circuitName
laps_race_pilot['desgasteCircuito'] = desgaste
laps_race_pilot['vueltasCircuito'] = vueltas_totales
laps_race_pilot['kmCircuitoTotales'] = km_total



#Ahora hay que meter las paradas, habria que añadir una columna con la siguiente parada, dependiendo de la vuelta en la que se encuentre
paradas_list = paradas
paradas_list


#En cada intervalo de ventana de parada, se añade la siguiente parada de acuerdo con el numero de vuelta en el que vamos
#Es decir, del intervalo 1-9, el valor de la siguiente parada es 10, del 10-30, el valor es 31, y así
#NOTA: Cuando se hace la ultima parada, ¿cual debe ser el valor del la siguiente parada? ¿El numero de fin de vuelta?


inserciones = 1
paradasList_insertar = []
for i in paradas_list:
        while (inserciones <= i):
            paradasList_insertar.append(i)
            inserciones = inserciones + 1
            if (inserciones == paradas_list[nParadas-1]):
                while inserciones <= nVueltasCircuito:
                    paradasList_insertar.append(nVueltasCircuito)
                    inserciones = inserciones + 1
paradasList_insertar
len(paradasList_insertar)
        

#Insertamos la nueva columna
laps_race_pilot['nextPit'] = paradasList_insertar
laps_race_pilot


# En principio, ahora tenemos el dataset para:
# 
# - Un piloto
# - En un gran premio
# - Durante todas sus vueltas


laps_race_pilot.to_csv("../datasets/laps_" + nombreCircuito + "_" + nombrePiloto  +"_" +anyoGp+ ".csv", index=False)
