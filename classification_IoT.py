# -->>>> A base de dados Utilizada no modelo está disponivél em: https://www.unb.ca/cic/datasets/iotdataset-2023.html

# Vamos importar alguma bibliotecas
import pandas as pd
# Comandos no sistema operacional
import os
# Função que emite avisos no python3
import warnings
warnings.filterwarnings('ignore') 
import keras
import tensorflow as tf
# Transformar os dados de texto em números
from sklearn.preprocessing import LabelEncoder
# Normalizar os dados
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils


def prepocess():
    # Vamos carregar a base de dados
    DATASET_DIRECTORY = 'IoT/CICIoT2023'
    # endswith('.csv') ----> verifica se termina com ".csv"
    df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
    # Realizando a ordenação dos dados
    df_sets.sort()
    #Dados de treinamento
    training_sets = df_sets[:int(len(df_sets)*.8)]
    #dados de teste
    test_sets = df_sets[int(len(df_sets)*.8):]
    normalizador_teste = MinMaxScaler(feature_range=(0,1))
    #dados de test
    for test in test_sets:
        path = DATASET_DIRECTORY+'/'+test
        test = pd.read_csv(path)
        classe_test = test['label']
        previssores_test = test.drop('label', axis=1)
    
    labelencoder_teste = LabelEncoder()
    classe_test = labelencoder_teste.fit_transform(classe_test)
    classe_test_dumpy = np_utils.to_categorical(classe_test)
    previssores_test = normalizador_teste.fit_transform(previssores_test)
    
    
    #dados de treino
    for train in training_sets:
        path = DATASET_DIRECTORY+'/'+train
        train = pd.read_csv(path)
        classe_train = train['label']
        previssores_train = train.drop('label', axis=1)
    
    #normalizar os dados entre 0 e 1
    #train /= 255
    normalizador_train = MinMaxScaler(feature_range=(0,1))
    labelencoder_train = LabelEncoder()
    classe_train = labelencoder_train.fit_transform(classe_train)
    classe_train_dumpy = np_utils.to_categorical(classe_train)
    previssores_train = normalizador_train.fit_transform(previssores_train)
    
    return previssores_train, classe_train_dumpy

#classe_train2 = np_utils.to_categorical(classe_train)

#--------> Rede <-------------------
# Camadas á definir pelo treinador da rede
# Adicionamos 3 camadas ocultas para ilustrar
def modelo(): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=100, activation='relu',kernel_initializer='normal', input_dim=46))
    model.add(tf.keras.layers.Dense(units=100, activation='relu', kernel_initializer='normal'))
    model.add(tf.keras.layers.Dense(units=100, activation='relu', kernel_initializer='normal'))
    model.add(tf.keras.layers.Dense(units=34, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


#função de callback
class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('categorical_accuracy') > 0.7970:
            print('categorical_accuracy: 76%, vamos cancelar o treino! Essa condição já nos satisfaz!')
            self.model.stop_training = True
            
#----------------->  treinamento <---------------------------
previssores_train,classe_train_dumpy = prepocess()
model = modelo()
resultado = model.fit(x=previssores_train, y=classe_train_dumpy, batch_size=50, epochs=10,verbose=1, callbacks=[Mycallback()])


