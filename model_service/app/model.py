import os
import numpy as np
import pandas as pd
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import logging as log

log.basicConfig(level=log.INFO, format='INFO:     %(asctime)s %(message)s')

MNIST_MODEL_PATH = "mnist_model.keras"
MNIST_MAX_VALUE = 255


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scales to [0, 1]
    x_train_normalized = x_train.astype('float32') / MNIST_MAX_VALUE
    x_test_normalized = x_test.astype('float32') / MNIST_MAX_VALUE

    # Adds depth for maps
    x_train_expanded = np.expand_dims(x_train_normalized, -1)
    x_test_expanded = np.expand_dims(x_test_normalized, -1)

    log.info('x_train shape: %s', x_train_expanded.shape)
    log.info('y_train shape: %s', y_train.shape)
    log.info('train samples: %s', x_train_expanded.shape[0])
    log.info('test samples: %s', x_test_expanded.shape[0])

    return (x_train_expanded, y_train), (x_test_expanded, y_test)


def init_model(x_train, y_train):
    try:
        log.info('Attempting to load trained model')
        model = keras.saving.load_model(MNIST_MODEL_PATH, compile=True)
    except ValueError:
        log.warning('Failed to load trained model')

        log.info('Building model')
        model = build_model(x_train, y_train)

    return model


def build_model(x_train, y_train):
    log.info('Building')
    model = build_untrained_model()

    log.info('Compiling')
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='acc'),
        ],
    )
    log.info('Training')
    trained_model = train_model(model, x_train, y_train)

    trained_model.save(MNIST_MODEL_PATH)

    return trained_model


def build_untrained_model():
    num_classes = 10
    input_shape = (28, 28, 1)
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax'),
        ]
    )


def train_model(model, x_train, y_train):
    batch_size = 128
    epochs = 20
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath='model_at_epoch_{epoch}.keras'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
    ]
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
    )

    return model


class TrainedMnistModel:

    def __init__(self):
        log.info('Loading MNIST dataset')
        (x_train, y_train), (x_test, y_test) = load_mnist()

        log.info('\n %s', pd.DataFrame(x_train[0][:, :, 0]))

        log.info('Initializing model')
        model = init_model(x_train, y_train)
        log.info(model.summary())

        self.__model = model
        self.__x_test = x_test
        self.__y_test = y_test
        self.__score = None

    @property
    def score(self):
        if self.__score is None:
            log.info('Evaluating')
            self.__score = self.evaluate()

            log.info('Final score: %s', self.__score)

        return self.__score

    def predict(self, features):
        return self.__model.predict(features)

    def evaluate(self):
        metrics = self.__model.metrics_names
        values = self.__model.evaluate(self.__x_test, self.__y_test, verbose=0)

        score = ", ".join(["{}: {}".format(metric, value) for metric, value in zip(metrics, values)])

        return score
