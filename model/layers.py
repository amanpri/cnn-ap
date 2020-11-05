from __future__ import annotations
  
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from model.base import Layer

class Convolution2D(Layer):

    def __init__(
        self, w: np.array,
        b: np.array,
        padding: str = 'valid',
        stride: int = 1
    ):
        """
        :param w -  4D tensor with shape (h_f, w_f, c_f, n_f)
        :param b - 1D tensor with shape (n_f, )
        :param padding - flag describing type of activation padding valid/same
        :param stride - stride along width and height of input volume
        ------------------------------------------------------------------------
        h_f - height of filter volume
        w_f - width of filter volume
        c_f - number of channels of filter volume
        n_f - number of filters in filter volume
        """
        self._w, self._b = w, b
        self._padding = padding
        self._stride = stride
        self._dw, self._db = None, None
        self._a_prev = None

    @classmethod
    def initialize(
        cls, filters: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid',
        stride: int = 1
    ) -> Convolution2D:
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(w=w, b=b, padding=padding, stride=stride)

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        if self._dw is None or self._db is None:
            return None
        return self._dw, self._db

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 4D tensor with shape (n, h_in, w_in, c)
        :output 4D tensor with shape (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        self._a_prev = np.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        n, h_in, w_in, _ = a_prev.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f

                output[:, i, j, :] = np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    self._w[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        return output + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 4D tensor with shape (n, h_out, w_out, n_f)
        :output 4D tensor with shape (n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self._a_prev.shape
        h_f, w_f, _, _ = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=self._a_prev, pad=pad)
        output = np.zeros_like(a_prev_pad)

        self._db = da_curr.sum(axis=(0, 1, 2)) / n
        self._dw = np.zeros_like(self._w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._w[np.newaxis, :, :, :, :] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self._dw += np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

        self._dw /= n
        kk=(self._dw-w_in)/self._dw*100
        print(np.abs(np.sum(kk)))
        
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]

    def set_wights(self, w: np.array, b: np.array) -> None:
        """
        :param w -  4D tensor with shape (h_f, w_f, c_f, n_f)
        :param b - 1D tensor with shape (n_f, )
        ------------------------------------------------------------------------
        h_f - height of filter volume
        w_f - width of filter volume
        c_f - number of channels of filter volume
        n_f - number of filters in filter volume
        """
        self._w = w
        self._b = b

    def calculate_output_dims(
        self, input_dims: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        :param input_dims - 4 element tuple (n, h_in, w_in, c)
        :output 4 element tuple (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self._w.shape
        if self._padding == 'same':
            return n, h_in, w_in, n_f
        elif self._padding == 'valid':
            h_out = (h_in - h_f) // self._stride + 1
            w_out = (w_in - w_f) // self._stride + 1
            return n, h_out, w_out, n_f
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )

    def calculate_pad_dims(self) -> Tuple[int, int]:
        """
        :output - 2 element tuple (h_pad, w_pad)
        ------------------------------------------------------------------------
        h_pad - single side padding on height of the volume
        w_pad - single side padding on width of the volume
        """
        if self._padding == 'same':
            h_f, w_f, _, _ = self._w.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self._padding == 'valid':
            return 0, 0
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )

    @staticmethod
    def pad(array: np.array, pad: Tuple[int, int]) -> np.array:
        """
        :param array -  4D tensor with shape (n, h_in, w_in, c)
        :param pad - 2 element tuple (h_pad, w_pad)
        :output 4D tensor with shape (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        h_pad - single side padding on height of the volume
        w_pad - single side padding on width of the volume
        """
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )

'''
class Maxpooling2D:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = int((W - self.pool)/self.s + 1)
        new_height = int((H - self.pool)/self.s + 1)
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(int(W/self.s)):
                for h in range(int(H/self.s)):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, w/self.pool, h/self.pool]
        return dx

    def extract(self):
        return 
 '''   
class FullyConnected(Layer):

    def __init__(self, w: np.array, b: np.array):
        """
        :param w - 2D weights tensor with shape (units_curr, units_prev)
        :param b - 1D bias tensor with shape (1, units_curr)
        ------------------------------------------------------------------------
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        self._w, self._b = w, b
        self._dw, self._db = None, None
        self._a_prev = None

    @classmethod
    def initialize(cls, units_prev: int, units_curr: int) -> DenseLayer:
        """
        :param units_prev - positive integer, number of units in previous layer
        :param units_curr - positive integer, number of units in current layer
        """
        w = np.random.randn(units_curr, units_prev) * 0.1
        b = np.random.randn(1, units_curr) * 0.1
        return cls(w=w, b=b)

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        if self._dw is None or self._db is None:
            return None
        return self._dw, self._db

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 2D tensor with shape (n, units_prev)
        :output - 2D tensor with shape (n, units_curr)
        ------------------------------------------------------------------------
        n - number of examples in batch
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        self._a_prev = np.array(a_prev, copy=True)
        return np.dot(a_prev, self._w.T) + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 2D tensor with shape (n, units_curr)
        :output - 2D tensor with shape (n, units_prev)
        ------------------------------------------------------------------------
        n - number of examples in batch
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        n = self._a_prev.shape[0]
        self._dw = np.dot(da_curr.T, self._a_prev) / n
        self._db = np.sum(da_curr, axis=0, keepdims=True) / n
        return np.dot(da_curr, self._w)

    def set_wights(self, w: np.array, b: np.array) -> None:
        """
        :param w - 2D weights tensor with shape (units_curr, units_prev)
        :param b - 1D bias tensor with shape (1, units_curr)
        ------------------------------------------------------------------------
        units_prev - number of units in previous layer
        units_curr -  number of units in current layer
        """
        self._w = w
        self._b = b

class Flatten(Layer):

    def __init__(self):
        self._shape = ()

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output - 1D tensor with shape (n, 1)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        self._shape = a_prev.shape
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 1D tensor with shape (n, 1)
        :output - ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        return da_curr.reshape(self._shape)

class ReLu(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        self._z = np.maximum(0, a_prev)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - ND tensor with shape (n, ..., channels)
        :output ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        dz = np.array(da_curr, copy=True)
        dz[self._z <= 0] = 0
        return dz

class Softmax(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 2D tensor with shape (n, k)
        :output 2D tensor with shape (n, k)
        ------------------------------------------------------------------------
        n - number of examples in batch
        k - number of classes
        """
        e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
        self._z = e / np.sum(e, axis=1, keepdims=True)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 2D tensor with shape (n, k)
        :output 2D tensor with shape (n, k)
        ------------------------------------------------------------------------
        n - number of examples in batch
        k - number of classes
        """
        return da_curr
