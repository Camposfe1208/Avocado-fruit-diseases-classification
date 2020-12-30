### Importar paqueterías ###
import keras
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing import image
from keras import optimizers
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from keras import backend as K

### Interrumpir sesiones de Keras ###
K.clear_session()

### Leer y mostrar las etiquetas de las imágenes ###
train = pd.read_csv ("/Avocado labels 1.csv")
train.head()
### Observar las columnas de identificación ###
train.columns

###                         CARGAR MODELO ENTRENADO                     ###
### SOLAMENTE EJECUTAR CUANDO YA SE TIENE EL MODELO Y LOS PESOS PARA EVITAR ENTRENAR DE NUEVO ###
modelo = ('C:/Modelos y pesos/modelo.h5')
pesos_modelo = ('/Modelos y pesos/pesos.h5')
model = load_model(modelo)
model.load_weights(pesos_modelo)

### Leer todas las imágenes, pre procesar a 300 x 300 px y los valores de los pixeles se encuentren del 0 - 1 ###
### Se vectorizan todas las imágenes ###
### Se define "X" (datos de entrenamiento) ###
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img("C:/Users/ulyss/Pictures/Avocado NBG/"+train['Identification'][i]+'.jpg',target_size=(300,300,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

### Revisar las nuevas características ###
X.shape

### Mostrar una imagen y su etiqueta correspondiente; el 9 significa el número de imagen a mostrar en el directorio ###
plt.imshow(X[9])
train['Condition'][9]

### Se define "y" (etiquetas de identificación) y se vectorizan todas las etiquetas ###
y = np.array(train.drop(['Identification','Condition'],axis = 1))
y.shape

### Se definen los datos de entrenamiento y de validación ###
### El 80 % serán los datos de entrenamiento y el 30 % de validación ###
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

### Definición de la arquitectura del modelo ###
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation="relu", input_shape=(300,300,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.30))

model.add(Dense(3, activation='softmax'))

### Revisar la arquitectura hecha ###
model.summary()

### Compilar el modelo ###
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

### Entrenar el modelo ###
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=128, verbose = 1)

### Graficar precisión y pérdida ###
def plot_LearningCurve(history,epoch):
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Precisión del Modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Épocas')
    plt.legend(['Entrenamiento', 'Validación'], loc = 'upper left')
    plt.show()
    ###error###
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Épocas')
    plt.legend(['Entrenamiento', 'Validación'], loc = 'upper left')
    plt.show()
    
plot_LearningCurve(history, 25)

### Crear una matriz de confusión para datos de validación ###
y_pred = model.predict_classes(X_test)
print(y_pred)
target_names = ['class 0(Healthy)','class 1(Scab)', 'class 2(Anthracnose)']
print(classification_report(np.argmax(y_test, axis = 1), y_pred, target_names = target_names))
print(confusion_matrix(np.argmax(y_test, axis = 1), y_pred))

### Graficar la matriz de confusión utilizando los datos generados arriba ###
array1 = [[302, 5, 29],
         [10, 229, 13],
         [83, 16, 110]]
df_cm = pd.DataFrame(array1, index = ["Healthy", "Scab", "Anthracnose"],
                  columns = ["Healthy", "Scab", "Anthracnose"])
plt.figure(figsize = (12,7))
plt.title('Matriz de Confusión para Datos de Validación', fontsize = 20)
plt.ylabel("True labels", fontsize = 20)
plt.xlabel("Predicted labels", fontsize = 20)

tick_marks = np.arange(len(array1))
plt.xticks(tick_marks, array1, rotation = 0)
plt.yticks(tick_marks, array1, rotation = 90)

sn.heatmap(df_cm, annot=True, cmap="Reds", annot_kws = {"size":20}, fmt = "0.0f")

### Guardar los pesos y el modelo del entrenamiento ###
dir = ('C:/Users/ulyss/Documents/Modelos y pesos14')
if not os.path.exists(dir):
    os.mkdir(dir)
    model.save('/Modelos y pesos/modelo.h5')
    model.save_weights('/Modelos y pesos/pesos.h5')

### Leer y evaluar imágenes con el modelo hecho ###
img = image.load_img("/Avocado NBG/Avocado 0475.jpg", target_size = (300,300,3))
img = image.img_to_array(img)
img = img/255.0

classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,300,300,3))
top_3 =np.argsort(proba[0])[:-4:-1]
for i in range(3):
    print("{}".format(classes1[top_3[i]])+ "({:.3})".format(proba[0][top_3[i]]))
    

plt.imshow(img)
