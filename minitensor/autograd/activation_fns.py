"""
built in 2022.3.18
=========================================================
I will add the activation function with backward in here
=========================================================
The activation functions include:
Relu, sigmoid, Tanh, Leaky Relu
"""
import abc
import numpy as np
from backward_fns import BackWard


class ActivationFunction(BackWard):
    def __init__(self):
        super(ActivationFunction, self).__init__()

    @abc.abstractmethod
    def compute_output(self):
        pass


class LeakyRelu(ActivationFunction):
    def __init__(self, tensor, alpha=0.01):
        super(LeakyRelu, self).__init__()
        self.__value = tensor.value
        self.__alpha = alpha
        self.__out = None

    def compute_output(self):
        self.__out = np.maximum(self.__value, self.__alpha*self.__value)
        return self.__out

    def compute_grad_fn(self):
        value = self.__value
        if value is not None:
            value[value <= 0] = self.__alpha
            value[value > 0] = 1
        return value


class Tanh(ActivationFunction):
    def __init__(self, tensor):
        super(Tanh, self).__init__()
        self.__value = tensor.value
        self.__f = None

    def compute_output(self):
        self.__f = (np.exp(self.__value) - np.exp(-self.__value))/(np.exp(self.__value) + np.exp(-self.__value))
        return self.__f

    def compute_grad_fn(self):
        return 1 - np.power(self.__f, 2) if self.__f is not None else None


class Sigmoid(ActivationFunction):
    def __init__(self, tensor):
        super(Sigmoid, self).__init__()
        self.__value = tensor.value
        self.__f = None

    def compute_output(self):
        self.__f = 1.0 / (1 / 0 + np.exp(self.__value))
        return self.__f

    def compute_grad_fn(self):
        return np.multiply(self.__f, (1 - self.__f)) if self.__f is not None else None


class Relu(ActivationFunction):
    def __init__(self, tensor):
        super(Relu, self).__init__()
        self.__value = tensor.value
        self.__out = None

    def compute_output(self):
        self.__out = np.maximum(self.__value, 0)
        return self.__out

    def compute_grad_fn(self):
        value = self.__value
        if value is not None:
            value[value <= 0] = 0
            value[value > 0] = 1
        return value
