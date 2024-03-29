"""
Begin Date: 2022.3.6
Author: ChuQi Zhang
Last Change Date: 2022.3.6
=================================================
We add backward function in this file
1. compute_grad_fn:
   compute the gradient for the tensor
2. backward_grad_fn:
   update the gradient for the children of tensor
=================================================
"""
import abc
import numpy as np


class BackWard:
    def __init__(self):
        super(BackWard, self).__init__()

    @abc.abstractmethod
    def compute_grad_fn(self):
        pass


class AddBackWard(BackWard):
    def __init__(self, tensor):
        super(AddBackWard, self).__init__()
        self.value = tensor

    def compute_grad_fn(self):
        l_grad, r_grad = np.ones((1, 1), dtype=np.float), np.ones((1, 1), dtype=np.float)
        return l_grad, r_grad

    def __str__(self):
        return "AddBackWard"


class SubBackWard(BackWard):
    def __init__(self, tensor):
        super(SubBackWard, self).__init__()
        self.value = tensor

    def compute_grad_fn(self):
        l_grad, r_grad = np.ones((1, 1), dtype=np.float), -1 * np.ones((1, 1), dtype=np.float)
        return l_grad, r_grad

    def __str__(self):
        return "SubBackWard"


class MulBackWard(BackWard):
    def __init__(self, tensor):
        super(MulBackWard, self).__init__()
        self.value = tensor

    def compute_grad_fn(self):
        l_grad, r_grad = np.dot(self.value.grad, self.value.r_children.value.T), \
                         np.dot(self.value.l_children.value.T, self.value.grad)
        return l_grad, r_grad

    def __str__(self):
        return "MulBackWard"


class DivBackWard(BackWard):
    def __init__(self, tensor):
        super(DivBackWard, self).__init__()
        self.value = tensor

    def compute_grad_fn(self):
        l_grad, r_grad = np.divide(1.0, self.value.r_children), \
                         -(np.divide(self.value.l_children, np.power(self.value.r_children, 2)))
        return l_grad, r_grad

    def __str__(self):
        return "DivBackWard"


class PowBackWard(BackWard):
    def __init__(self, tensor):
        super(PowBackWard, self).__init__()
        self.value = tensor

    def compute_grad_fn(self):
        l_grad, r_grad = self.value.r_children * np.power(self.value.l_children, self.value.r_children), None
        return l_grad, r_grad

    def __str__(self):
        return "PowBackWard"

