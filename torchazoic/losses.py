from torch import Tensor


class Loss:
    def loss(self, pred: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

    def grad(self, pred: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    def loss(self, pred: Tensor, actual: Tensor) -> Tensor:
        abs_error = (pred - actual.float()).pow(2)
        return abs_error.sum()

    def grad(self, pred: Tensor, actual: Tensor) -> Tensor:
        abs_error_grad = 2 * (pred - actual.float())
        return abs_error_grad
