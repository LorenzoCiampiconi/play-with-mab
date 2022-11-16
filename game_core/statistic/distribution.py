import abc

import numpy as np
from numpy.random import normal, binomial
from numpy.random import beta
from scipy import stats


class DistributionABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self) -> float:
        pass

    @property
    def expected_value(self) -> float:
        pass

    def get_cumulate_expected_value_for_steps(self, steps):
        cumulated_expected_values = []
        cumulated_expected_value = 0

        for _ in range(steps):
            cev = cumulated_expected_value + self.expected_value
            cumulated_expected_values.append(cev)
            cumulated_expected_value = cev

        return cumulated_expected_values


class GaussianDistribution(DistributionABC):
    def __init__(self, mean, std_dev):
        self._std_dev = std_dev
        self._mean = mean

    def sample(self) -> float:
        return normal(loc=self._mean, scale=self._std_dev)

    @property
    def expected_value(self) -> float:
        return self._mean


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

    @property
    def expected_value(self) -> float:
        return self.a / (self.a + self.b)


class BernoulliDistribution(DistributionABC):
    def __init__(self, p, seed=None):
        self._p = p
        self.rdstate = np.random.RandomState(seed) if seed is not None else None

    @property
    def p(self):
        return self._p

    def sample(self) -> float:
        return self.rdstate.binomial(1, self._p, 1)[0] if self.rdstate is not None else binomial(1, self._p, 1)[0]

    @property
    def expected_value(self) -> float:
        return self._p
