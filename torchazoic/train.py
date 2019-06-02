from torch import Tensor
from torchazoic.model import Model
from torchazoic.losses import Loss, MSE
from torchazoic.optimizers import Optimizer, SGD
from torchazoic.utils import DataIterator, BatchIterator


def train(
    model: Model,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 5000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD(),
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            pred = model.forward(batch.inputs)
            epoch_loss += loss.loss(pred, batch.targets).item()
            grad = loss.grad(pred, batch.targets)
            model.backward(grad)
            optimizer.step(model)
        print(epoch, epoch_loss)
