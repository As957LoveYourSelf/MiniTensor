"""
Begin DATE: 2021.11.8
Author: ZhangChuQi

Problem:
1. 当使用numpy数组对Node类型进行运算操作时，会出现类似以下问题:
        [<__main__.Node object at 0x000001FA336264C0>
        <__main__.Node object at 0x000001FA33473DF0>
        <__main__.Node object at 0x000001FA334738E0>]
    即无法显示运算后的结果。初步推测和numpy.ndarray内部的运算符
    重载操作有关。
2. 围绕以上问题，MiniTensor目前只支持Tensor之间以及Tensor与int、float之间的运算操作。
    但可以将ndarray转化为Tensor

Usage:
Node类只提供基本的算数运算，且不考虑更多细节的矩阵运算。
若要查看矩阵运算用法，请参考Tensor类

TODO:
1. You can change the content in __str__.
2. If you want, you can make the program compatible ndarray type to operation
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
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    def __radd__(self, other):
        if isinstance(other, Node):
            return other.__add__(self.value)
        elif isinstance(other, (int, float)):
            return self.__add__(other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    # mul part
    """
    """
    def __mul__(self, other):
        if isinstance(other, Node):
            return Node(self.value * other.value)
        elif isinstance(other, (int, float)):
            return Node(self.value * other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    def __rmul__(self, other):
        if isinstance(other, Node):
            return other.__mul__(self.value)
        elif isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    # div part
    """
    """
    def __truediv__(self, other):
        if isinstance(other, Node):
            return Node(self.value / other.value)
        elif isinstance(other, (int, float)):
            return Node(self.value / other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    def __rdiv__(self, other):
        if isinstance(other, Node):
            return other.__truediv__(self.value)
        elif isinstance(other, (int, float)):
            return self.__truediv__(other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    def __floordiv__(self, other):
        if isinstance(other, Node):
            return Node(self.value // other.value)
        elif isinstance(other, (int, float)):
            return Node(self.value // other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    def __rfloordiv__(self, other):
        if isinstance(other, Node):
            return other.__floordiv__(self.value)
        elif isinstance(other, (int, float)):
            return self.__floordiv__(other)
        else:
            raise TypeError("Your operation value must be type from(int, float, Tensor)")

    # pow part
    """
    """
    def __pow__(self, power, modulo=None):
        return Node(pow(self.value, power, mod=modulo))

    def __str__(self):
        _str = f"[MiniTensor]=>Tensor: \n{self.value}"
        return _str


