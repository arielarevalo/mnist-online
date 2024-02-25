import os
import numpy as np
import pandas as pd
import logging as log

os.environ['KERAS_BACKEND'] = 'torch'
import keras

log.basicConfig(level=log.INFO, format='INFO:     %(asctime)s %(message)s')

MODEL_DIR_PATH = 'models/'
MNIST_MODEL_PATH = MODEL_DIR_PATH + 'mnist.keras'
MNIST_MAX_VALUE = 255


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_processed = preprocess_input(x_train)
    x_test_processed = preprocess_input(x_test)

    log.info('x_train shape: %s', x_train_processed.shape)
    log.info('y_train shape: %s', y_train.shape)
    log.info('train samples: %s', x_train_processed.shape[0])
    log.info('test samples: %s', x_test_processed.shape[0])

    return (x_train_processed, y_train), (x_test_processed, y_test)


def preprocess_input(raw_input):
    # Scales to [0, 1]
    input_normalized = raw_input.astype('float32') / MNIST_MAX_VALUE

    # Adds depth for maps
    return np.expand_dims(input_normalized, -1)


def validate_input(plain_matrices):
    """
    Converts a list of plain Python matrices to a NumPy array and validates that each is a 28x28 matrix with all positive numerical values.

    :param plain_matrices: The input list of plain Python matrices to validate and convert. Expected shape is (n, 28, 28) where n is the number of samples.
    :type plain_matrices: list
    :returns: A NumPy array if the input is valid, otherwise raises a ValueError.
    :rtype: np.ndarray
    :raises ValueError: If the input is not a list of 28x28 matrices, or any matrix contains non-positive values.
    """
    # Convert the list of plain Python matrices to a NumPy array
    matrices = np.array(plain_matrices)

    # Check if the conversion results in a numpy array with the expected dimensions
    if len(matrices.shape) != 3 or matrices.shape[1] != 28 or matrices.shape[2] != 28:
        raise ValueError("Each input matrix must be 28x28 in size.")

    # Check if all elements are positive numerical values
    if not np.all(matrices >= 0):
        raise ValueError("All elements in the matrices must be positive.")

    return matrices


def init_model(x_train, y_train):
    try:
        log.info('Attempting to load trained model')
        model = keras.saving.load_model(MNIST_MODEL_PATH, compile=True)
        log.info(model.summary())
    except ValueError:
        log.warning('Failed to load trained model')

        log.info('Building trained model')
        model = build_model(x_train, y_train)

    return model


def build_model(x_train, y_train):
    log.info('Building untrained model')
    model = build_untrained_model()
    log.info(model.summary())

    log.info('Compiling model')
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='acc'),
        ],
    )
    log.info('Training model')
    trained_model = train_model(model, x_train, y_train)

    trained_model.save(MNIST_MODEL_PATH)

    return trained_model


def build_untrained_model():
    num_classes = 10
    input_shape = (28, 28, 1)
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),  # Outputs (28, 28, 1)

            # Each Conv2D layer slides a (3, 3, n) kernel two-dimensionally over the feature volume, performing a
            # three-dimensional Hadamard product between the "feature volume" and the kernel, and summing all positions
            # in the resulting matrix, before running that through the activation function. We can see that the output
            # per filter is then a two-dimensional feature map.
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Outputs (26, 26, 64)
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Outputs (24, 24, 64)

            # Picks the max value for each (2, 2) area in each two-dimensional feature map
            keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Outputs (12, 12, 64)

            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),  # Outputs (10, 10, 128)
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),  # Outputs (8, 8, 128)

            # Computes the average value for each feature map
            keras.layers.GlobalAveragePooling2D(),  # Outputs (128,)

            # Sets 0.5 of values to zero, scales remainder by 1 / (1 - 0.5)
            keras.layers.Dropout(0.5),  # Same output,

            # The Dense layer has an (n, num_classes) weight matrix W. We multiply this by the input vector X like so
            # Z = X * W + B, where B is a vector of biases for each class. Z has our value for each class.
            # We're multiplying each input value by an entire column of weights for each class, and getting back a
            # vector of class-specific values. We then use an activation function to obtain normalized probabilities
            # for each class based on the initial scores for each class.
            keras.layers.Dense(num_classes, activation='softmax'),  # Outputs (num_classes,)
        ]
    )


def train_model(model, x_train, y_train):
    batch_size = 128
    epochs = 20
    callbacks = [
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

    def predict(self, feature_sets):
        """
        Calculates a series of predictions for a series of input feature sets.

        :param feature_sets: A series of input features
        :return: A series of predictions
        """
        validated_feature_sets = validate_input(feature_sets)

        processed_feature_sets = preprocess_input(validated_feature_sets)

        return self.__model.predict(processed_feature_sets)

    def evaluate(self):
        metrics = self.__model.metrics_names
        values = self.__model.evaluate(self.__x_test, self.__y_test, verbose=0)

        score = ", ".join(["{}: {}".format(metric, value) for metric, value in zip(metrics, values)])

        return score
