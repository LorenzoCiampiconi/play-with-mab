import abc

import numpy as np
from numpy.random import normal, binomial
from numpy.random import beta
from scipy import stats


class DistributionABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self) -> float:
        pass


class GaussianDistribution(DistributionABC):
    def __init__(self, mean, std_dev):
        self._std_dev = std_dev
        self._mean = mean

    def sample(self) -> float:
        return normal(loc=self._mean, scale=self._std_dev)


class PositiveGaussianDistribution(GaussianDistribution):
    def sample(self) -> float:
        sample = super().sample()

        return sample if sample >= 0 else 0


class BetaDistribution(DistributionABC):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self) -> float:
        return beta(a=self.a, b=self.b)

    def pdf(self, x):
        return stats.beta.pdf(x, self.a, self.b)


class BernoulliDistribution(DistributionABC):
    def __init__(self, p, seed=None):
        self._p = p
        self.rdstate = np.random.RandomState(seed) if seed is not None else None

    @property
    def p(self):
        return self._p

    def sample(self) -> float:
        return self.rdstate.binomial(1, self._p, 1)[0] if self.rdstate is not None else binomial(1, self._p, 1)[0]
