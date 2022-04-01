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
2. 围绕以上问题，MiniTensor将ndarray转化为Tensor
3. Graph built
4. More then one Gradients calculation: Gradients accumulation
5. check mul operator
=====================================================================
|   2021.11.12 update: the problem 1,2 have been solved             |
=====================================================================
-----------------------------------------------------------------------
2022.3.18:
Complete the autograd module
=================================================================
Usage:
Every father node number in Tensor must be 2
-----------------------------------------------------------------
Node类只提供基本的算数运算，且不考虑更多细节的矩阵运算。
若要查看矩阵运算用法，请参考Tensor类
=================================================================
TODO:
1. You can change the content in __str__.
2. add grad calculate program.
"""
import numpy as np
from minitensor.core import graph as g
from minitensor.autograd.backward_fns import MulBackWard, AddBackWard, SubBackWard, DivBackWard


class Node(object):
    def __init__(self, value, require_grads, grad_fn):
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
        if isinstance(value, int):
            value = [value]
        if not isinstance(value, np.ndarray):
            value = np.array(value, np.float16)
        self.value = value
        self.father = []
        self.l_children = None
        self.r_children = None
        self.main_children = None
        self.node_name = 'GraphNode' + str(g.nodes_num)
        g.nodes_name.append(self.node_name)
        g.nodes_num += 1
        self.grad_op = grad_fn if require_grads else None
        self.activation_function = None
        self.is_leafNode = True
        self.is_root_Node = True
        self.requires_grad = require_grads
        self.grad = 1.0 if self.requires_grad else 0

    def backward(self):
        if self.requires_grad is False:
            raise RuntimeError("Unable compute gradient, try set the requires_grad True.")
        if self.is_leafNode:
            return
        l_grad, r_grad = self.grad_op(self).compute_grad_fn()
        if self.l_children is not None:
            if self.l_children.requires_grad:
                self.l_children.grad += l_grad

        if self.r_children is not None:
            if self.r_children.requires_grad:
                self.r_children.grad += r_grad
        self.l_children.backward()
        self.r_children.backward()

    def zero_grad(self):
        if self.is_leafNode:
            return
        self.grad = 0
        if self.l_children is not None:
            self.l_children.zero_grad()
        if self.r_children is not None:
            self.r_children.zero_grad()

    def __rename(self, new_name):
        if new_name in g.nodes_name:
            raise UserWarning("Your name was in the node name list, we will delete it. We hope try another name next "
                              "time")
        try:
            g.nodes_name.remove(self.node_name)
            self.node_name = new_name
        except ValueError:
            self.node_name = new_name
        g.nodes_name.append(self.node_name)

    def __required_grad_fn(self, other, v):
        if other is None:
            self.father.append(v)
        else:
            g.graph.add_nodes(key=other.node_name, value=other.father, type='forward')
            g.graph.add_nodes(key=self.node_name, value=self.father, type='forward')

            other.father.append(v)
            other.is_root_Node = False

            v.r_children = other

        v.l_children = self
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
        v = Tensor(np.add(self.value, other.value), self.requires_grad, grad_fn=AddBackWard)
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
        v = Tensor(self.value - other.value, self.requires_grad, grad_fn=SubBackWard)
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
            v = Tensor(np.multiply(self.value, other.value), self.requires_grad, grad_fn=MulBackWard)
        else:
            v = Tensor(np.dot(self.value, other.value), self.requires_grad, grad_fn=MulBackWard)
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
            v = Tensor(np.divide(self.value, other), self.requires_grad, grad_fn=DivBackWard)
        else:
            v = Tensor(np.dot(self.value, np.linalg.inv(other.value)), self.requires_grad, grad_fn=DivBackWard)
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
        _str = f"[MiniTensor]:Tensor({str(self.value)}', {'grad_op = '+str(self.grad_op) if self.requires_grad else ' '})"
        return _str


class Tensor(Node):
    """
    include some operations
    """

    def __init__(self, value: (int, list, tuple, np.ndarray), requires_grad=False, name=None, *args, **kwargs):
        super(Tensor, self).__init__(value, require_grads=requires_grad, grad_fn=kwargs.get("grad_fn"))
        if name is not None:
            assert isinstance(name, str)
            self.__rename(name)

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

    def T(self):
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
