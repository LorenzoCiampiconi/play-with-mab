import abc

from numpy.random import normal


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
