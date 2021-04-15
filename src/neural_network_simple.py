from numpy import loadtxt
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout

# load the dataset
dataset = loadtxt('csv_files/nn_input.csv', delimiter=',')
# split into input (x) and output (y) variables
x = dataset[:,0:25]
y = dataset[:,25]

y = np_utils.to_categorical(y, 3)

model = Sequential()
model.add(Dense(64, input_dim=25, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.summary()

# compile the keras model
#early_stop = EarlyStopping(monitor='accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(x, y, epochs=200, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f%%' % (accuracy*100))
