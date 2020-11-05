import numpy as np
import time


def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    :param x - features array with (n, ...) shape
    :param y - one hot ground truth array with (n, k) shape
    :batch_size - number of elements in single batch
    ----------------------------------------------------------------------------
    n - number of examples in data set
    k - number of classes
    """
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[0])), axis=0)
        )


def format_time(start_time: time.time, end_time: time.time) -> str:
    """
    :param start_time - beginning of time period
    :param end_time - ending of time period
    :output - string in HH:MM:SS.ss format
    ----------------------------------------------------------------------------
    HH - hours
    MM - minutes
    SS - seconds
    ss - hundredths of a second
    """
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def convert_prob2categorical(probs: np.array) -> np.array:
    """
    :param probs - softmax output array with (n, k) shape
    :return categorical array with (n, ) shape
    ----------------------------------------------------------------------------
    n - number of examples
    k - number of classes
    """
    return np.argmax(probs, axis=1)

def convert_prob2one_hot(probs: np.array, slice_size: int):
    """
    :param probs - softmax output array with (n, k) shape
    :return one hot array with (n, k) shape
    ----------------------------------------------------------------------------
    n - number of examples
    k - number of classes
    """
    if probs.shape==(slice_size,):
        return probs
    else:
        class_idx = convert_prob2categorical(probs)
        one_hot_matrix = np.zeros_like(probs)
        one_hot_matrix[np.arange(probs.shape[0]), class_idx] = 1
        return one_hot_matrix

def softmax_accuracy(y_hat: np.array, y: np.array, slice_size: int):
    """
    :param y_hat - 2D one-hot prediction tensor with shape (n, k)
    :param y - 2D one-hot ground truth labels tensor with shape (n, k)
    ----------------------------------------------------------------------------
    n - number of examples in batch
    k - number of classes
    """
    y_hat = convert_prob2one_hot(y_hat, slice_size)
    if y_hat.shape == (slice_size,):
        return (y_hat == y).all().mean()
    else:
        return (y_hat == y).all(axis=1).mean()


def softmax_cross_entropy(y_hat, y, eps=1e-20) -> float:
    """
    :param y_hat - 2D one-hot prediction tensor with shape (n, k)
    :param y - 2D one-hot ground truth labels tensor with shape (n, k)
    ----------------------------------------------------------------------------
    n - number of examples in batch
    k - number of classes
    """
    n = y_hat.shape[0]
    return - np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / n
