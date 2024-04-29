from itertools import islice, cycle
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from avalanche.evaluation import GenericPluginMetric, PluginMetric, Metric
from avalanche.evaluation.metric_definitions import TResult
from avalanche.evaluation.metric_results import MetricValue, LoggingType
import numpy as np

from utils.visualisation import plot_Gauss2D_contour


class ClusterHistograms(Metric[Figure]):
    """Create the cluster histogram plot."""

    def __init__(self):
        self.cluster_assignments = None

    def result(self, **kwargs) -> Optional[TResult]:
        """
        Obtains the value of the metric.

        :return: The value of the metric.
        """
        f = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.matshow(self.cluster_assignments)
        plt.title("Cluster Assignments")
        f.show()
        return f

    def reset(self, **kwargs) -> None:
        """
        Resets the metric internal state.

        :return: None.
        """
        pass

    def update(self, cluster_assignments):
        self.cluster_assignments = cluster_assignments.detach().cpu().numpy()


class ClusterHistogramsP:
    """Update ClusterHistogram plot after each training epoch."""
    def __init__(self):
        self.metric = ClusterHistograms()
        self.last_val = None

    def result(self):
        return self.last_val

    def after_training_epoch(self, strategy):
        self.metric.update(strategy.model.classifier.N_k_for_label)
        x = strategy.clock.train_iterations
        self.last_val = self.metric.result()
        mval = MetricValue(self, 'cluster_assignments', self.last_val, x,
                           LoggingType.FIGURE)
        self.last_val = mval
        return mval


class Gaussian2DPlot(Metric[Figure]):
    """Create the cluster histogram plot."""

    def __init__(self, scenario):
        """.

        :param X: GMM input data (numpy array)
        :param y: target labels (numpy array)
        """
        self.model = None

        self.X = []
        self.y = []
        for eid, exp in enumerate(scenario.train_stream):
            Xe, ye = exp.dataset[:][0].numpy(), exp.dataset[:][1].numpy()
            self.X.append(Xe)
            self.y.append(ye)
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        self.colors = np.array(list(islice(cycle(
            ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
             "#984ea3", "#999999", "#e41a1c", "#dede00"]),
            int(10), )))

    def result(self, **kwargs) -> Optional[TResult]:
        """
        Obtains the value of the metric.

        :return: The value of the metric.
        """
        f = plt.figure()
        ax = plt.subplot(1, 1, 1)

        # get emission learned params
        em = self.model.get_expected_distributions()['emissions']

        # plot data
        ax.scatter(self.X[:, 0], self.X[:, 1], s=10, color=self.colors[self.y])
        for i in range(self.model.K):
            if self.model.N_k[i] > 0:
                plot_Gauss2D_contour(mu=em['mean'][i].numpy(),
                                     Sigma=em['cov'][i].numpy(),
                                     ax_handle=ax)
        return f

    def reset(self, **kwargs) -> None:
        """
        Resets the metric internal state.

        :return: None.
        """
        pass

    def update(self, model):
        self.model = model


class Gaussian2DPlotP:
    """Update ClusterHistogram plot after each training epoch."""
    def __init__(self, scenario):
        self.metric = Gaussian2DPlot(scenario)
        self.last_val = None

    def result(self):
        return self.last_val

    def after_training_epoch(self, strategy):
        self.metric.update(strategy.model.classifier)
        x = strategy.clock.train_iterations
        self.last_val = self.metric.result()
        mval = MetricValue(self, 'gaussian_2d_plot', self.last_val, x,
                           LoggingType.FIGURE)
        self.last_val = mval
        return mval


if __name__ == '__main__':
    cluster_assignments = np.random.randn(100, 10)

    f = plt.figure()
    f.show()
