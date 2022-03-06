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
from core.graph import nodes_names, nodes_num, graph


class Node(object):
    def __init__(self, value, require_grads):
        """
        In here, We keep the value is the type of numpy.ndarray
        include all the node bases
        value:
        father:
        grad:
        grad_op:
        is_leafNode:
        =========================================================
        :param value:
        :param require_grads:
        =========================================================
        """
        if not isinstance(value, (int, list, tuple, np.ndarray)):
            raise TypeError(f"You must apply type where from (int, list, tuple, numpy.ndarray), not {type(value)}")
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float16)
        self.value = value
        self.father = []
        self.children = []
        self.node_name = 'GraphNode' + str(nodes_num)
        self.grad = None
        self.grad_op = None
        self.is_leafNode = True
        self.is_root_Node = True
        self.requires_grad = require_grads

    def __rename(self, new_name):
        if new_name in nodes_names:
            raise UserWarning("Your name was in the node name list, we will delete it. We hope try another name next "
                              "time")
        try:
            nodes_names.remove(self.node_name)
            self.node_name = new_name
        except ValueError:
            self.node_name = new_name
        nodes_names.append(self.node_name)

    def __required_grad_fn(self, other, v):
        if other is None:
            self.father.append(v)
        else:
            graph.add_nodes(key=other.node_name, value=other.father, type='forward')
            graph.add_nodes(key=self.node_name, value=self.father, type='forward')

            other.father.append(v)
            other.is_root_Node = False

            v.children.append(other)

        v.children.append(self)
        v.is_leafNode = False

        self.is_root_Node = False
        self.father.append(v)

    # add part
    def __add__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            pass
        elif isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other, requires_grad=self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        v = Tensor(np.add(self.value, other.value), self.requires_grad)
        self.__required_grad_fn(other, v)
        return v

    def __sub__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            pass
        elif isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other, requires_grad=self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        v = Tensor(self.value - other.value, self.requires_grad)
        self.__required_grad_fn(other, v)
        return v

    def __mul__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            pass
        elif isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other, requires_grad=self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        if other.shape() == self.value.shape:
            v = Tensor(np.multiply(self.value, other.value), self.requires_grad)
        else:
            v = Tensor(np.dot(self.value, other.value), self.requires_grad)
        self.__required_grad_fn(other, v)
        return v

    def __truediv__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Tensor):
            pass
        elif isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other, requires_grad=self.requires_grad)
        else:
            raise TypeError(f"Your operation value must be type from(int, float, Tensor), not {type(other)}")
        if other.shape() == self.value.shape:
            v = Tensor(np.multiply(self.value, np.linalg.inv(other.value)), self.requires_grad)
        else:
            v = Tensor(np.dot(self.value, np.linalg.inv(other.value)), self.requires_grad)
        self.__required_grad_fn(other, v)
        return v

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
        v = Tensor(pow(self.value, power, mod=modulo), self.requires_grad)
        self.__required_grad_fn(None, v)
        return v

    def __str__(self):
        """

        :return:
        """
        _str = f"[MiniTensor]:Tensor({self.value}{(', ', 'grad_op=' + self.grad_op) if self.grad is not None else ''})"
        return _str


class Tensor(Node):
    """
    include some operations
    """

    def __init__(self, value, requires_grad=True, name=None, *args, **kwargs):
        super(Tensor, self).__init__(value, require_grads=requires_grad)
        if name is not None:
            assert isinstance(name, str)
            self.__rename(name)

    def __call__(self, n_n=nodes_num, *args, **kwargs):
        n_n += 1

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

    # backward function
    def backward(self):
        if self.is_root_Node:
            pass
        else:
            raise ValueError("You should make the value be only output")

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
