"""Replayable dataset choices, sample order and augmentation seeds."""

import numpy as np
from torch.utils.data import Sampler


class MixtureBatchSampler(Sampler):
    """One dataset per global batch, sharded across ranks without duplicate samples.

    All order/augmentation decisions depend only on the recipe and step, so a
    resume does not depend on worker prefetch state or filesystem iteration order.
    """

    def __init__(self, lengths, probabilities, batch_size, rank, world_size, seed, steps, start=0, smoke=False):
        self.lengths = lengths
        self.probabilities = probabilities
        self.batch_size = batch_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.steps, self.start, self.smoke = seed, steps, start, smoke
        if not 0 <= rank < world_size or min(lengths) < batch_size * world_size:
            raise ValueError("Each dataset must contain at least one complete global batch.")

    def __len__(self):
        return self.steps - self.start

    def __iter__(self):
        choice = np.random.RandomState(self.seed)
        generators = [np.random.RandomState(self.seed + 1000 + i) for i in range(len(self.lengths))]
        orders = [g.permutation(n) for g, n in zip(generators, self.lengths)]
        cursors = [0] * len(orders)
        global_batch = self.batch_size * self.world_size
        for step in range(self.steps):
            dataset = step % len(orders) if self.smoke else int(choice.choice(len(orders), p=self.probabilities))
            if cursors[dataset] + global_batch > self.lengths[dataset]:
                orders[dataset] = generators[dataset].permutation(self.lengths[dataset])
                cursors[dataset] = 0
            offset = cursors[dataset] + self.rank * self.batch_size
            indices = orders[dataset][offset:offset + self.batch_size]
            cursors[dataset] += global_batch
            if step >= self.start:
                yield [(dataset, int(index), self.seed + step * global_batch + self.rank * self.batch_size + j)
                       for j, index in enumerate(indices)]
