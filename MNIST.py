# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras


# %%
import matplotlib.pyplot as plt


# %%
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D


# %%
mnist = tf.keras.datasets.mnist #reading the dataset


# %%
(x_train, y_train), (x_test,y_test) = mnist.load_data() #loading train and test data


# %%
print(x_train[7])

plt.imshow(x_train[7], cmap=plt.cm.binary) 
plt.show()


# %%
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# %%
#Reshaping x_train and x_test as cnn expects (batch, height, width, channels)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape)


# %%
#Converting matrix values to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train[7])


# %%
#normalize
x_train = x_train/255
x_test = x_test/255

print(x_train[7])


# %%
#One hot encoding to transform integers to binary matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test,10)
print(y_train.shape)
print(y_test.shape)


# %%
#Building the model

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation = 'relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3,3), padding='same', activation = 'relu'))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.50))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation ='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))


# %%
model.summary()


# %%
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
             )
          


# %%
history = model.fit(x_train, y_train, batch_size=32, epochs = 3, validation_split=0.4)


# %%
test_score = model.evaluate(x_test, y_test, batch_size=128, verbose=2)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1]*100.0, "%")


# %%
import matplotlib.pyplot as plt


# %%
#plot accuracy per epoch
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# %%
# plot accuracy per epoch
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# %%
model.save('my_model')
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model('my_model')


# %%
test_scores = model.evaluate(x_test, y_test, batch_size = 128, verbose = 0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1]*100.0)


# %%
image_index = 39
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, x_train.shape[1], x_train.shape[2], 1))
print("The output is of image_index", image_index, "is:", pred.argmax())


# %%
pred1 = model.predict(x_test, verbose=2)
y_predict = np.argmax(pred1, axis=1)

#Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

 
for cm in range(10):
    print(cm, confusion_matrix(np.argmax(y_test,axis=1),y_predict)[cm].sum())
con_matrix = confusion_matrix(np.argmax(y_test,axis=1),y_predict)
print(con_matrix)


#Normalizing the values
con_matrix = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]


# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(con_matrix, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})
plt.show()


# %%
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# %%
print(classification_report(np.argmax(y_test,axis=1),y_predict,target_names=labels))


# %%



