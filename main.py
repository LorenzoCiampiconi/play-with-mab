from distribution import GaussianDistribution
from mab import MabProblem
g_1 = GaussianDistribution(0,1)
g_2 = GaussianDistribution(0.5,2)
g_3 = GaussianDistribution(4,10)
g_4 = GaussianDistribution(3,5)
mab_problem = MabProblem({1: g_1, 2: g_2, 3: g_3, 4: g_4})