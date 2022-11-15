import abc

from numpy.random import normal, binomial


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


class BernoulliDistribution(DistributionABC):
    def __init__(self, p):
        self._p = p

    @property
    def p(self):
        return self._p

    def sample(self) -> float:
        return binomial(1, self._p, 1)[0]
