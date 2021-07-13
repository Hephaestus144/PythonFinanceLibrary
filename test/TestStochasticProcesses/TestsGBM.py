import numpy as np
import unittest
from src.Finance.StochasticProcesses.GBM import GBM


class GBM_Tests(unittest.TestCase):
    # driftless test
    # pfe type test
    # stats tests
    def test_deterministic(self):
        S0 = 100
        mu = 0.1
        sigma = 0.0
        dt = 0.1
        T = 1
        simulation_count = 1
        sp = GBM(S0, mu, sigma, T, dt, simulation_count)
        sp.generate_paths()
        # print(f"Final value was {sp.paths[0, -1]:.2f}")
        self.assertAlmostEqual(sp.paths[0, -1], 110.52, 2)

    def test_driftless(self):
        S0 = 100
        mu = 0.0
        sigma = 0.2
        dt = 0.1
        T = 1
        simulation_count = 100
        sp = GBM(S0, mu, sigma, T, dt, simulation_count)
        sp.generate_paths()
        # print(f"Final value was {sp.paths[0, -1]:.2f}")
        stddev = 100*(np.exp(sigma**2) - 1)
        avg = np.average(sp.paths[:, -1])
        self.assertTrue(100 - stddev < avg < 100 + stddev)

    # def test_plot(self):
    #     S0 = 100
    #     mu = 0.1
    #     sigma = 0.2
    #     dt = 0.1
    #     T = 1
    #     simulation_count = 100
    #     sp = GBM(S0, mu, sigma, dt, T, simulation_count)
    #     sp.generate_paths()
    #     sp.plot_paths()
