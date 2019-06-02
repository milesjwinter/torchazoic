from typing import Iterator, NamedTuple
import torch
from torch import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(
        self, inputs: Tensor, targets: Tensor
    ) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        if self.shuffle:
            rand_idxs = torch.randperm(inputs.shape[0])
            inputs = inputs.index_select(0, rand_idxs)
            targets = targets.index_select(0, rand_idxs)

        for batch_inputs, batch_targets in zip(
            inputs.split(self.batch_size), targets.split(self.batch_size)
        ):
            yield Batch(batch_inputs, batch_targets)
