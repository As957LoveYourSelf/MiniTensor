"""
Begin Date: 2021.11.15
Author: ChuQi Zhang
Last Change Date: 2021.11.18
"""
import numpy as np
from core.tensor import Tensor

class Graph:
    def __init__(self, root_node):
        super(Graph, self).__init__()
        if isinstance(root_node, Tensor):
            self.root_node = root_node
        else:
            raise TypeError("")

    def __call__(self, *args, **kwargs):
        pass

    def gradient_calculation(self, *values, grad_fn=None):
        """

        :param values:
        :param grad_fn:
        :return:
        """
        assert (isinstance(grad_fn, str))
        l_node, r_node, node_grad = values
        grad_fn = grad_fn.lower()
        if "add" in grad_fn:
            self.__add_gradient_calculate(*values)
        elif "sub" in grad_fn:
            self.__sub_gradient_calculate(*values)
        elif "mul" in grad_fn:
            self.__mul_gradient_calculate(*values)
        elif "div" in grad_fn:
            self.__div_gradient_calculate(*values)
        t_ones = node_grad if node_grad is not None else np.ones_like(node_grad)
        if isinstance(l_node, Tensor):
            for node in l_node.father.items():
                assert (isinstance(node, Tensor))
                if node.l_child is l_node:
                    pass
                elif node.r_child is l_node:
                    pass
                else:
                    raise RuntimeError("")
        else:
            return

        if isinstance(r_node, Tensor):
            pass
        else:
            return

    def show_graph(self):
        pass

    def __node_ergodic(self, root):
        """
        only Tensor class can be ergodic
        :param root:
        :return:
        """
        if isinstance(root, Tensor):
            if root.requires_grad:
                root.grad = Tensor(np.ones_like(root.value))
            self.gradient_calculation(root.l_child, root.r_child, root.grad, grad_fn=root.grad_op)
            self.__node_ergodic(root.l_child)
            self.__node_ergodic(root.r_child)
            return root
        return

    def __add_gradient_calculate(self, node):
        """

        :param l_node:
        :param r_node:
        :param node_grad:
        :return:
        """
        pass


    def __sub_gradient_calculate(self, l_node, r_node, node_grad):
        """

        :param l_node:
        :param r_node:
        :param node_grad:
        :return:
        """
        pass

    def __mul_gradient_calculate(self, l_node, r_node, node_grad):
        """

        :param l_node:
        :param r_node:
        :param node_grad:
        :return:
        """
        pass

    def __div_gradient_calculate(self, l_node, r_node, node_grad):
        """

        :param l_node:
        :param r_node:
        :param node_grad:
        :return:
        """
        pass

    def __mean_gradient_calculate(self, l_node, r_node, node_grad):
        """

        :param l_node:
        :param r_node:
        :param node_grad:
        :return:
        """
        pass


