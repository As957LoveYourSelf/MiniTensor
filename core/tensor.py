"""
Begin DATE: 2021.11.8
Last Change: 2021.11.15
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
4. More then one Gradients calculation: Gradients accumulation

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
    def __init__(self, value, require_grads):
        """
        In here, We keep the value is the type of numpy.ndarray
        Variables:
        value:
        father:
        grad:
        grad_op:
        is_leafNode:
        l_child:
        r_child:
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
        self.father = {}
        self.backward_n = {
            'add': 0,
            'sub': 0,
            'mul': 0,
            'div': 0
        }
        self.grad = None
        self.grad_op = None
        self.is_leafNode = False
        self.l_child = None
        self.r_child = None
        self.requires_grad = require_grads

    # add part
    def __add__(self, other):
        """

        :param other:
        :return:
        """
        op_name = f"AddBackward{self.backward_n['add']}"
        if isinstance(other, Tensor):
            self.father[op_name] = Tensor(np.add(self.value, other.value), self.requires_grad)
            other.backward_n['add'] += 1
        elif isinstance(other, (int, float)):
            self.father[op_name] = Tensor(np.add(self.value, other), self.requires_grad)
        elif isinstance(other, np.ndarray):
            self.father[op_name] = Tensor(np.add(self.value, other), self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        if self.requires_grad:
            self.father[op_name].r_child, self.father[op_name].l_child = other, self
            self.father[op_name].grad_op = op_name
        self.is_leafNode = True
        self.backward_n['add'] += 1
        return self.father[op_name]

    def __sub__(self, other):
        """

        :param other:
        :return:
        """
        op_name = f"AddBackward{self.backward_n['sub']}"
        if isinstance(other, Tensor):
            self.father[op_name] = Tensor(self.value - other.value, self.requires_grad)
            other.backward_n['sub'] += 1
        elif isinstance(other, (int, float)):
            self.father[op_name] = Tensor(self.value - other, self.requires_grad)
        elif isinstance(other, np.ndarray):
            self.father[op_name] = Tensor(np.subtract(self.value, other), self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        if self.requires_grad:
            self.father[op_name].l_child, self.father[op_name].r_child = self, other
            self.father[op_name].grad_op = op_name
        self.is_leafNode = True
        self.backward_n['sub'] += 1
        return self.father[op_name]

    def __mul__(self, other):
        """

        :param other:
        :return:
        """
        op_name = f"AddBackward{self.backward_n['mul']}"
        if isinstance(other, Tensor):
            if other.shape() == self.value.shape:
                self.father[op_name] = Tensor(np.multiply(self.value, other.value), self.requires_grad)
            else:
                self.father[op_name] = Tensor(np.dot(self.value, other.value), self.requires_grad)
            other.backward_n['mul'] += 1
        elif isinstance(other, (int, float)):
            self.father[op_name] = Tensor(self.value * other, self.requires_grad)
        elif isinstance(other, np.ndarray):
            if np.shape(other) == np.shape(self.value):
                self.father[op_name] = Tensor(np.multiply(self.value, other), self.requires_grad)
            else:
                self.father[op_name] = Tensor(np.dot(self.value, other), self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        if self.requires_grad:
            self.father[op_name].l_child, self.father[op_name].r_child = self, other
            self.father[op_name].grad_op = op_name
        self.is_leafNode = True
        self.backward_n['mul'] += 1
        return self.father[op_name]

    def __truediv__(self, other):
        """

        :param other:
        :return:
        """
        op_name = f"AddBackward{self.backward_n['div']}"
        if isinstance(other, Tensor):
            if other.shape() == self.value.shape:
                self.father[op_name] = Tensor(np.multiply(self.value, np.linalg.inv(other.value)), self.requires_grad)
            else:
                self.father[op_name] = Tensor(np.dot(self.value, np.linalg.inv(other.value)), self.requires_grad)
            other.backward_n['div'] += 1
        elif isinstance(other, (int, float)):
            self.father[op_name] = Tensor(self.value / other, self.requires_grad)
        elif isinstance(other, np.ndarray):
            self.father[op_name] = Tensor(np.multiply(self.value, np.linalg.inv(other)), self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        if self.requires_grad:
            self.father[op_name].l_child, self.father[op_name].r_child = self, other
            self.father[op_name].grad_op = op_name
        self.is_leafNode = True
        self.backward_n['div'] += 1
        return self.father[op_name]

    def __floordiv__(self, other):
        """

        :param other:
        :return:
        """
        raise UserWarning("You should keep your tensor type to be float, instant of int")

    def __pow__(self, power, modulo=None):
        """

        :param power:
        :param modulo:
        :return:
        """
        self.father = Tensor(pow(self.value, power, mod=modulo), self.requires_grad)
        self.father.grad_op = "pow" if self.requires_grad else None
        self.father.l_child = self
        return self.father

    def __str__(self):
        """

        :return:
        """
        _str = f"[MiniTensor]:Tensor({self.value}{(', ','grad_op='+self.grad_op) if self.grad is not None else ''})"
        return _str


class Tensor(Node):
    def __init__(self, value, requires_grad=False, name=None):
        super(Tensor, self).__init__(value, require_grads=requires_grad)
        self.name = name

    def __calculate_gradient(self, grad_fn):
        pass

    def rank(self):
        return np.linalg.matrix_rank(self.value)

    def det(self):
        """

        :return:
        """
        return np.linalg.det(self.value)

    def inv(self):
        if self.value.shape[-1] != self.value.shape[-2]:
            raise TypeError("")
        return Tensor(np.linalg.inv(self.value))

    def backward(self):
        pass

    def t(self):
        return Tensor(self.value.T)

    def add(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(np.add(self.value, tensor.value))
        else:
            raise TypeError(f"The other compute value must be Tensor, instead of {type(tensor)}")

    def mul(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(np.multiply(self.value, tensor.value))
        else:
            raise TypeError(f"The other compute value must be Tensor, instead of {type(tensor)}")

    def dot(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(np.dot(self.value, tensor.value))
        else:
            raise TypeError(f"The other compute value must be Tensor, instead of {type(tensor)}")

    def sub(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(np.subtract(self.value, tensor.value))
        else:
            raise TypeError(f"The other compute value must be Tensor, instead of {type(tensor)}")

    def div(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(np.dot(self.value, np.linalg.inv(tensor.value)))
        else:
            raise TypeError(f"The other compute value must be Tensor, instead of {type(tensor)}")

    def reshape(self, resize):
        """

        :param resize:
        :return:
        """
        return Tensor(np.reshape(self.value, resize))

    def shape(self):
        return np.shape(self.value)

    def size(self, dim=None):
        return np.size(self.value, axis=dim)

    def mean(self):
        return np.mean(self.value)

    def sum(self):
        return np.sum(self.value)

    def pow(self, exp, mod=None):
        return Tensor(pow(self.value, exp, mod=mod))
