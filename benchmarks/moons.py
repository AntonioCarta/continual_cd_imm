from collections.abc import Sequence
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from itertools import cycle, islice

from sklearn.utils import check_random_state

from avalanche.benchmarks import nc_benchmark, dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import DataAttribute


def make_moon(num_samples=100, center=(.0, .0), arc=(0, np.pi)):
    """Sample random points in an arc centered in center.

    :param num_samples:
    :param center: cartesian coordinates of the center.
    :param arc: start and end angles (in radians).
    :return: cartesian coordinates of the samples as a 2D arrays of shape
        <num_samples, coordinates>.
    """
    x = center[0] + np.cos(np.linspace(arc[0], arc[1], num_samples))
    y = center[1] + np.sin(np.linspace(arc[0], arc[1], num_samples))
    return np.vstack([x, y]).T


class MoreMoonsDataset:
    def __init__(self, num_samples=100, num_classes=2, noise=None,
                 random_state=None, arcs=None):
        """More Moons Dataset.

        Similar to scikit-learn moons dataset but adapted for
        multi-class, domain-incremental, and class-incremental CL scenarios.

        :param num_samples:
        :param num_classes:
        :param noise:
        :param random_state:
        """
        # make num_samples a multiple of num_classes because I'm too lazy to
        # manage the remainder.
        self.num_samples = num_samples // num_classes * num_classes
        self.num_classes = num_classes
        self.noise = noise

        samples_per_class = self.num_samples // num_classes

        generator = check_random_state(random_state)

        if arcs is None:
            arcs = [(0, np.pi), (-np.pi, 0)]
        xs = []
        ys = []
        curr_center = [.0, .0]
        for c in range(num_classes):
            curr_arc = arcs[c % len(arcs)]
            m = make_moon(samples_per_class, curr_center, curr_arc)

            new_x = 1. if c % 2 == 0 else .0
            delta_y = 0.5 if c % 2 == 0 else 1.0
            curr_center[0] = curr_center[0] + new_x
            curr_center[1] = curr_center[1] + delta_y

            xs.append(m)
            ys.append(np.ones(samples_per_class, dtype=np.intp) * c)

        X = np.concatenate(xs, axis=0)
        y = np.hstack(ys)

        if noise is not None:
            X += generator.normal(scale=noise, size=X.shape)

        mean = np.mean(X, axis=0, keepdims=True)
        X = (X-mean)#/(num_classes/2)
        self.X = torch.tensor(X, dtype=torch.float)
        self.targets = torch.tensor(y)

    def __getitem__(self, item):
        return self.X[item], self.targets[item]

    def __len__(self):
        return self.X.shape[0]


def plot_moons(X, y, ax=None):
    if ax is None:
        ax = plt.subplot()

    plt.title("Moons", size=18)

    colors = np.array(list(islice(cycle(
        ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
         "#984ea3", "#999999", "#e41a1c", "#dede00"]),
         int(max(y) + 1),)))

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    return ax


def NCMoons(
    n_experiences: int,
    num_samples=100,
    num_classes=2,
    noise=None,
    *,
    return_task_id=False,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = False,
):
    train_data = MoreMoonsDataset(num_samples, num_classes, noise)
    test_data = MoreMoonsDataset(num_samples, num_classes, noise)

    tgs = DataAttribute(train_data.targets, "targets")
    train_data = TensorDataset(train_data.X, train_data.targets)
    train_data = AvalancheDataset([train_data], data_attributes=[tgs])

    tgs = DataAttribute(test_data.targets, "targets")
    test_data = TensorDataset(test_data.X, test_data.targets)
    test_data = AvalancheDataset([test_data], data_attributes=[tgs])

    return nc_benchmark(
        train_dataset=train_data,
        test_dataset=test_data,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=False)


def NIMoons(n_experiences: int, num_samples=100, num_classes=2, noise=None):
    train_s = []
    test_s = []
    for i in range(n_experiences):
        delta_arc = np.pi / n_experiences
        curr_arcs = [
            (i*delta_arc, (i + 1) * delta_arc),
            (-np.pi + i*delta_arc, -np.pi + (i + 1)*delta_arc)
        ]

        train_data = MoreMoonsDataset(num_samples, num_classes, noise, arcs=curr_arcs)
        test_data = MoreMoonsDataset(num_samples, num_classes, noise, arcs=curr_arcs)

        train_dataset = TensorDataset(train_data.X, train_data.targets)
        tgs = DataAttribute(train_data.targets, "targets")
        doms = DataAttribute([i for _ in train_data.targets], "domains")
        train_dataset = AvalancheDataset([train_dataset], data_attributes=[tgs, doms])
        test_dataset = TensorDataset(test_data.X, test_data.targets)
        tgs = DataAttribute(test_data.targets, "targets")
        doms = DataAttribute([i for _ in test_data.targets], "domains")
        test_dataset = AvalancheDataset([test_dataset], data_attributes=[tgs, doms])

        train_s.append(train_dataset)
        test_s.append(test_dataset)

    return dataset_benchmark(train_datasets=train_s, test_datasets=test_s)


if __name__ == '__main__':
    np.random.seed(0)

    n_samples = 1500
    data = MoreMoonsDataset(n_samples, 8, 0.05)
    plot_moons(data.X.numpy(), data.targets.numpy())

    # class-incremental adds new classes over time.
    # check experiences by plotting their data.
    benchmark = NCMoons(5, 1000, 10, 0.05)

    plt.figure()
    colors = np.array(list(islice(cycle(
        ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
         "#984ea3", "#999999", "#e41a1c", "#dede00"]),
         int(10),)))
    for eid, exp in enumerate(benchmark.train_stream):
        plt.subplot(1, 5, eid + 1)
        X, y = exp.dataset[:][0].numpy(), exp.dataset[:][1].numpy()
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])

        plt.xlim(-1.5, 5.5)
        plt.ylim(-1, 6)

    plt.title("Class Incremental")
    plt.show()

    # domain-incremental where the moon's arc is incrementally discovered
    # over time. Importantly, the shift is gradual and not random.
    # check experiences by plotting their data.
    benchmark = NIMoons(5, 1000, 10, 0.05)

    plt.figure()
    colors = np.array(list(islice(cycle(
        ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
         "#984ea3", "#999999", "#e41a1c", "#dede00"]),
         int(10),)))

    xs, ids = [], []
    for eid, exp in enumerate(benchmark.train_stream):
        X = exp.dataset[:][0].numpy()
        xs.append(X)
        ids.append(eid + np.zeros(X.shape[0], dtype=np.intp))
    X = np.concatenate(xs, axis=0)
    e = np.concatenate(ids, axis=0)

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[e])
    plt.title("Domain Incremental")
    plt.show()
