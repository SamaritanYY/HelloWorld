
'''using keras and mnist to train BP network'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import keras


(x_train,y_train),(x_test,y_test)=mnist.load_data()
#initialize data shape
x_train=x_train.reshape(x_train.shape[0],-1)/255.0
x_test=x_test.reshape(x_test.shape[0],-1)/255.0
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

model = Sequential()
model.add(Dense(output_dim=30, input_dim=784, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10)

cost, accuracy = model.evaluate(x_test, y_test, batch_size=100)
print("test cost: ", cost)
print("test accuracy: ", accuracy)
