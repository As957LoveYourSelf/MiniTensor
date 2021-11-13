"""
Begin DATE: 2021.11.8
Last Change: 2021.11.12
Author: ZhangChuQi
=============================================================
What I do in this code:
-------------------------------------------------------------
1. define Class <Node>, where main save gradient and children.
2. define Class <Tensor>, which is the main calculation unit.
==============================================================

Problem(Solved):
--------------------------------------------------------------
1. 当使用numpy数组对Node类型进行运算操作时，会出现类似以下问题:
        [<__main__.Tensor object at 0x000001FA336264C0>
        <__main__.Tensor object at 0x000001FA33473DF0>
        <__main__.Tensor object at 0x000001FA334738E0>]
    即无法显示运算后的结果。初步推测和numpy.ndarray内部的运算符
    重载操作有关。
2. 围绕以上问题，MiniTensor目前只支持Tensor之间以及Tensor与int、float之间的运算操作。
    但可以将ndarray转化为Tensor

=====================================================================
|   2021.11.12 update: the problem 1,2 have been solved             |
=====================================================================
3. Graph built


-----------------------------------------------------------------------

=================================================================
Usage:
-----------------------------------------------------------------
Node类只提供基本的算数运算，且不考虑更多细节的矩阵运算。
若要查看矩阵运算用法，请参考Tensor类
=================================================================
TODO:
1. You can change the content in __str__.
2. add grad calculate program.
"""
import numpy as np


class Node(object):
    def __init__(self, value, require_grads=False):
        """
        In here, We keep the value is the type of numpy.ndarray
        Variables:
        self.value:
        self.father:
        self.grad:
        self.grad_op:
        self.is_leafNode:
        self.l_child:
        self.r_child:
        =========================================================
        :param value:
        :param require_grads:
        =========================================================
        """
        if not isinstance(value, (int, list, tuple, np.ndarray)):
            raise TypeError(f"You must apply type where from (int, list, tuple, numpy.ndarray), not {type(value)}")
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)
        self.value = value
        self.father = None
        self.grad = None
        self.grad_op = None
        self.is_leafNode = False
        self.l_child = None
        self.r_child = None

    # add part
    def __add__(self, other):
        """

        :param other:
        :return:
        """
        self.grad_op = "add"
        if isinstance(other, Tensor):
            other.grad_op = "add"
            self.father = Tensor(np.add(self.value, other.value))
        elif isinstance(other, (int, float)):
            self.father = Tensor(np.add(self.value, other))
        elif isinstance(other, np.ndarray):
            self.father = Tensor(np.add(self.value, other))
        else:
            self.grad_op = None
            other.grad_op = None
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        return self.father

    def __radd__(self, other):
        """

        :param other:
        :return:
        """
        self.grad_op = "add"
        if isinstance(other, Tensor):
            other.grad_op = "add"
            self.father = other.__add__(self.value)
        elif isinstance(other, (int, float)):
            self.father = self.__add__(other)
        elif isinstance(other, np.ndarray):
            self.father = Tensor(np.array(np.add(other, self.value)))
        else:
            self.grad_op = None
            other.grad_op = None
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        return self.father

    def __sub__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            return Tensor(self.value - other.value)
        elif isinstance(other, (int, float)):
            return Tensor(self.value - other)
        elif isinstance(other, np.ndarray):
            return Tensor(np.subtract(self.value, other))
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")

    def __rsub__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            return Tensor(other.value - self.value)
        elif isinstance(other, (int, float)):
            return Tensor(other - self.value)
        elif isinstance(other, np.ndarray):
            return Tensor(np.subtract(other, self.value))
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")

    # mul part
    def __mul__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            return Tensor(np.multiply(self.value, other.value))
        elif isinstance(other, (int, float)):
            return Tensor(self.value * other)
        elif isinstance(other, np.ndarray):
            return Tensor(np.array(np.multiply(self.value, other)))
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")

    def __rmul__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            return other.__mul__(self.value)
        elif isinstance(other, (int, float)):
            return self.__mul__(other)
        elif isinstance(other, np.ndarray):
            return Tensor(np.array(np.multiply(other, self.value)))
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")

    # div part
    def __truediv__(self, other):
        """

        :param other:
        :return:
        """

        if isinstance(other, Tensor):
            return Tensor(np.array(np.multiply(self.value, other.value.I)))
        elif isinstance(other, (int, float)):
            return Tensor(self.value / other)
        elif isinstance(other, np.ndarray):
            return Tensor(np.array(np.multiply(self.value, np.linalg.inv(other))))
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")

    def __rdiv__(self, other):
        """

        :param other:
        :return:
        """

        if isinstance(other, Tensor):
            return other.__truediv__(self.value)
        elif isinstance(other, (int, float, np.ndarray)):
            return Tensor(np.array(np.multiply(other, np.linalg.inv(self.value))))
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")

    def __floordiv__(self, other):
        """

        :param other:
        :return:
        """
        raise UserWarning("You should keep your tensor type to be float, instant of int")

    def __rfloordiv__(self, other):
        """

        :param other:
        :return:
        """
        raise UserWarning("You should keep your tensor type to be float, instant of int")

    # pow part
    def __pow__(self, power, modulo=None):
        """

        :param power:
        :param modulo:
        :return:
        """
        return Tensor(pow(self.value, power, mod=modulo))

    def __str__(self):
        """

        :return:
        """
        _str = f"[MiniTensor]:Tensor({self.value})"
        return _str


class Tensor(Node):
    def __init__(self, value, requires_grad=False):
        super(Tensor, self).__init__(value, require_grads=requires_grad)
        self.T = self.t()

    def __calculate_gradient(self, grad_fn):
        pass

    def rank(self):
        pass

    def det(self):
        pass

    def inv(self):
        pass

    def backward(self):
        pass

    def t(self):
        return self.value.T

    def add(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.add(self.value, other.value))

    def mul(self):
        pass

    def dot(self):
        pass

    def sub(self):
        pass

    def div(self):
        pass

    def reshape(self):
        pass

    def shape(self):
        pass

    def show_graph(self):
        pass

    def pow(self):
        pass
