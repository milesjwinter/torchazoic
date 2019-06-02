from typing import Dict, Callable
import torch
from torch import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = torch.randn(input_size, output_size)
        self.params["b"] = torch.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        self.grads["b"] = grad.sum(dim=0)
        self.grads["w"] = self.inputs.t() @ grad
        return grad @ self.params["w"].t()


ActType = Callable[[Tensor], Tensor]


class Activation(Layer):
    def __init__(self, f: ActType, df: ActType) -> None:
        super().__init__()
        self.f = f
        self.df = df

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.df(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def dtanh(x: Tensor) -> Tensor:
    """
    dtanh = 1/cosh^2=(cosh^2-sinh^2)/cosh^2=1-tanh^2
    w/ hyperbolic identity: cosh^2-sinh^2=1
    """
    y = x.tanh()
    return 1 - y.pow(2)


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, dtanh)
