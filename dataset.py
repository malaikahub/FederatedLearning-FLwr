import tensorflow as tf
import numpy as np

def load_datasets(num_clients=5):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # split for clients
    client_data = []
    size = len(x_train) // num_clients

    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size
        client_data.append((x_train[start:end], y_train[start:end]))

    return client_data, (x_test, y_test)