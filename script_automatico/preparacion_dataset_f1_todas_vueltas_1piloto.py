#!/usr/bin/env python
# coding: utf-8

# In[3]:


import fastf1
import pandas as pd


# In[4]:


#Obtención de todos los circuitos en csv

circuitos = pd.read_csv("../circuits/circuits_2022.csv")
circuitos


# In[5]:


#Limpiamos el dataframe de circuitos
circuitos = circuitos.drop(['url', 'Location/lat', 'Location/long', 'Location/country'], axis = 1)
circuitos


# In[6]:


#Creamos un nuevo dataframe con el nombre, la localidad y el nivel de desgaste que pueden tener los neumaticos
circuitos_desgaste = circuitos.drop(['circuitId'], axis=1)
circuitos_desgaste


# In[7]:


#Creamos una lista con el nivel de desgaste, fuente PIRELLI F1 2022
desgaste = ['Bajo', 'Alto', 'Alto', 'Bajo', 'Alto', 'Bajo', 'Medio', 'Medio', 'Bajo', 'Bajo', 'Bajo', 'Medio', 'Medio', 'Medio', 'Medio', 'Bajo', 'Medio', 'Alto', 'Medio', 'Bajo', 'Medio', 'Medio']
circuitos_desgaste['desgaste'] = desgaste
circuitos_desgaste
vueltas_circuito = ['58', '56', '57', '51', '66', '70', '63', '71', '50', '61', '57', '78', '53', '71', '53', '71', '52', '44', '53', '70', '58', '72']
circuitos_desgaste['vueltas_totales'] = vueltas_circuito
circuitos_desgaste
km_circuito = ['307.574', '308.405', '308.238', '306.049', '307.104', '306.630', '309.049', '305.879', '308.450', '308.706', '308.326', '260.286', '306.720', '306.452', '309.690', '305.354', '306.198', '308.052', '307.471', '305.270', '305.355', '306.648']
circuitos_desgaste['km_total'] = km_circuito
circuitos_desgaste


# In[8]:


#Vamos entonces a crear un dataframe para un piloto en una determinada 
#carrera, por ejemplo, Fernando Alonso y Monaco 2022
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #Omitimos los future warning

fastf1.Cache.enable_cache('../cache') # Para cachear los datos y no tener que bajarlos todo el tiempo

#A continuacion, nos guardamos la sesión en una variable de tipo sesion

barcelona_race = fastf1.get_session(2022, 'Barcelona', 'R')
barcelona_race.load(weather=True, telemetry= True)




# In[9]:


#Nos interesa saber el numero de vueltas que dio Alonso en Barcelona 2022


laps_barcelona_alo = barcelona_race.laps.pick_driver("ALO")
weather = barcelona_race.weather_data # Cogemos tambien el tiempo que hizo al inicio de la carrera


# In[10]:


laps_barcelona_alo


# In[11]:


#Limpiamos el dataset con solo lo que nos interesa

laps_barcelona_alo = laps_barcelona_alo.filter(['Time', 'LapTime', 'LapNumber', 'Compound', 'TyreLife', 'FreshTyre', 'TrackStatus', 'Stint', 'PitInTime'])
laps_barcelona_alo


# In[12]:


#De aqui solo queremos la temperatura de la pista, la temperatura ambiente y si llovía o no 
#Importante, hacemos match de la columna time, o ponemos simplemente el tiempo que hacia en la salida????
weather = weather.filter(['Time', 'AirTemp', 'Rainfall', 'TrackTemp'])
weather


# In[13]:


weather['Time'] = weather['Time'].astype(str)
weather


# In[14]:


#Modificamos la columna de Time, para que solo nos muestre la hora y el minuto, esto es para
#que casen las keys al hacer el join y podamos tener info de la temperatura minuto a minuto en las vueltas
weather['Time'] = weather["Time"].apply(lambda x: x[7:12])
weather


# In[15]:


#Hacemos lo mismo para la columa Time de las vueltas
laps_barcelona_alo['Time'] = laps_barcelona_alo['Time'].astype(str)
laps_barcelona_alo['Time'] = laps_barcelona_alo["Time"].apply(lambda x: x[7:12])
laps_barcelona_alo


# In[16]:


#Hacemos el join para meter en el mismo dataset, la imformacion en cada vuelta de la temperatura ambiente y de la pista
laps_barcelona_alo = laps_barcelona_alo.set_index('Time').join(weather.set_index('Time'))


# In[17]:


#Ahora tenemos las 64 vueltas que dio Alonso y la temperatura que hacia, necesitamos:
# - Posicion de salida


# In[18]:


#Para la posicion de salida
posicion_salida = barcelona_race.get_driver("ALO")['GridPosition']
posicion_salida


# In[19]:


#Necesitamos ahora los outputs, que es la siguiente vuelta a parar
vueltas_parada = laps_barcelona_alo[laps_barcelona_alo['PitInTime'].notna()]
vueltas_parada


# In[20]:


nVuelta_parada = vueltas_parada['LapNumber']
nParadas = 0
paradas = []
for i in nVuelta_parada:
    paradas.append(i)
    nParadas = nParadas + 1


print("El numero de paradas ha sido: ", nParadas)

print("Las paradas han sido en las vueltas: ", paradas)


# In[21]:


#Eliminanos la variable de la hora a la que entra a box porque ya no es necesaria
laps_barcelona_alo.drop(['PitInTime'], axis = 1, inplace=True)


# In[22]:


laps_barcelona_alo


# In[23]:


#Ahora hay que añadir al dataframe las variables
# - Posicion de salida
# - Siguiente parada (dependera de la vuelta en la que se encuentre)
# - Todas variables del circuito en el que nos encontramos, en este caso Barcelona
# - No creo que haga falta el piloto, las estrategias son independientes del piloto (o eso voy a pensar... que no será asi)
circuito_actual = circuitos_desgaste.loc[circuitos_desgaste['Location/locality'] == 'Montmeló']
circuito_actual = circuito_actual.filter(['circuitName', 'desgaste', 'vueltas_totales', 'km_total'])
circuito_actual


# In[24]:


#Hay que hacer una funcion para sacar el numero de vueltas (nVueltas) que se ha dado en ese circuito, y repetir nVueltas para crear una lista con el valor
nVueltasCircuito=laps_barcelona_alo.tail(1).LapNumber.values[0]
circuitName = circuito_actual.circuitName.values[0]
desgaste = circuito_actual.desgaste.values[0]
km_total = circuito_actual.km_total.values[0]
vueltas_totales = circuito_actual.vueltas_totales.values[0]
desgaste

    


# In[25]:


#La salida se la tenemos que meter a todas las filas del dataframe
laps_barcelona_alo['posicionSalida'] = posicion_salida
laps_barcelona_alo['circuitName'] = circuitName
laps_barcelona_alo['desgasteCircuito'] = desgaste
laps_barcelona_alo['vueltasCircuito'] = km_total
laps_barcelona_alo['kmCircuitoTotales'] = vueltas_totales

laps_barcelona_alo


# In[26]:


#Ahora hay que meter las paradas, habria que añadir una columna con la siguiente parada, dependiendo de la vuelta en la que se encuentre
paradas_list = paradas
paradas_list


# In[27]:


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
        


# In[28]:


#Insertamos la nueva columna
laps_barcelona_alo['nextPit'] = paradasList_insertar
laps_barcelona_alo


# En principio, ahora tenemos el dataset para:
# 
# - Un piloto
# - En un gran premio
# - Durante todas sus vueltas

# In[29]:


laps_barcelona_alo.to_csv("../datasets/laps_barcelona_alo.csv", index=False)


# In[ ]:





# In[ ]:




