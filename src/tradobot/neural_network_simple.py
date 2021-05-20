"""Module containing the model training and validation function."""
from numpy import loadtxt
from keras.models import Sequential
from keras.utils import np_utils

# from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout

INPUT_FILENAME = "csv_files/nn_input.csv"


def train_and_test(input_filename):
    """Load the dataset.

    Then perform training and validation.
    """
    # load the dataset
    dataset = loadtxt(input_filename, delimiter=",")

    # split into input (x) and output (y) variables
    # training-validation split taken as 80:20
    num_classes = 3
    split = 0.8
    num_epochs = 200
    input_batch_size = 10
    first_dense = 64
    num_dense_layers = 3  # excluding final dense(3) layer for classification
    dropout_rate = 0.5
    include_batch_norm = True
    # include_early_stop = True

    validation_index = int(split * dataset.shape[0])

    x_train = dataset[0:validation_index, 0:-1]
    y_train = dataset[0:validation_index, -1]
    y_train = np_utils.to_categorical(y_train, num_classes)

    x_test = dataset[validation_index:, 0:-1]
    y_test = dataset[validation_index:, -1]
    y_test = np_utils.to_categorical(y_test, num_classes)

    num_neurons = first_dense  # x2 neurons each subsequent layer

    model = Sequential()
    for _ in range(num_dense_layers):
        model.add(
            Dense(num_neurons, input_dim=x_train.shape[1], activation="relu")
        )
        if include_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        num_neurons = num_neurons * 2

    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    # # compile the keras model
    # if include_early_stop:
    #     early_stop = EarlyStopping(
    #         monitor='accuracy',
    #         min_delta=0,
    #         patience=10,
    #         verbose=1,
    #         mode='auto',
    #         baseline=None,
    #         restore_best_weights=True
    #     )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=input_batch_size)

    # evaluate the keras model
    _, train_accuracy = model.evaluate(x_train, y_train)
    _, test_accuracy = model.evaluate(x_test, y_test)
    return train_accuracy, test_accuracy
    # print('Train Accuracy: %.2f%%' % (train_accuracy*100))
    # print('Test Accuracy: %.2f%%' % (test_accuracy*100))


if __name__ == "__main__":
    print(train_and_test(INPUT_FILENAME))
