"""
Begin DATE: 2021.11.8
Author: ZhangChuQi

"""
import numpy as np
import collections


class Node(object):
    def __init__(self, value, require_grads=True):
        if not isinstance(value, (int, list, tuple, np.ndarray)):
            raise TypeError("You must apply type where from (int, list, tuple, numpy.ndarray)")
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.child = []
        self.grad = None

    # add part
    """
    """
    def __add__(self, other):
        if isinstance(other, Node):
            return Node(self.value + other.value)
        elif isinstance(other, (int, float)):
            return Node(self.value + other)

    def __radd__(self, other):
        if isinstance(other, Node):
            return other.__add__(self.value)
        elif isinstance(other, (int, float)):
            return self.__add__(other)

    # mul part
    """
    """
    def __mul__(self, other):
        if isinstance(other, Node):
            return Node(self.value * other.value)
        elif isinstance(other, (int, float, np.ndarray)):
            return Node(self.value * other)

    def __rmul__(self, other):
        if isinstance(other, (Node, np.ndarray)):
            return other.__mul__(self.value)
        elif isinstance(other, (int, float)):
            return self.__mul__(other)

    # div part
    """
    """
    def __truediv__(self, other):
        if isinstance(other, Node):
            return Node(self.value / other.value)
        elif isinstance(other, (int, float, np.ndarray)):
            return Node(self.value / other)

    def __rdiv__(self, other):
        if isinstance(other, (Node, np.ndarray)):
            return other.__truediv__(self.value)
        elif isinstance(other, (int, float)):
            return self.__truediv__(other)

    def __floordiv__(self, other):
        if isinstance(other, Node):
            return Node(self.value // other.value)
        elif isinstance(other, (int, float, np.ndarray)):
            return Node(self.value // other)

    def __rfloordiv__(self, other):
        if isinstance(other, (Node, np.ndarray)):
            return other.__floordiv__(self.value)
        elif isinstance(other, (int, float, np.ndarray)):
            return self.__floordiv__(other)

    # pow part
    """
    """
    def __pow__(self, power, modulo=None):
        return Node(pow(self.value, power, mod=modulo))

    def __str__(self):
        _str = f"[MiniTensor]=>Tensor: [{self.value}]"
        return _str




