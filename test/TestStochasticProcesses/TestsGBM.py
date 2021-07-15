import numpy as np
import unittest
from src.StochasticProcesses.GBM import GBM


class TestsGBM(unittest.TestCase):
    # driftless test
    # pfe type test
    # stats tests
    def test_deterministic(self):
        initial_asset_value = 100
        drift = 0.1
        volatility = 0.0
        time_step_size = 0.1
        time_to_maturity = 1
        simulation_count = 1
        gbm = GBM(initial_asset_value, drift, volatility, time_to_maturity, time_step_size, simulation_count)
        gbm.generate_paths()
        self.assertAlmostEqual(gbm.paths[0, -1], 110.52, 2)

    def test_driftless(self):
        initial_asset_value = 100
        drift = 0.0
        volatility = 0.2
        time_step_size = 0.1
        time_to_maturity = 1
        simulation_count = 100
        gbm = GBM(initial_asset_value, drift, volatility, time_to_maturity, time_step_size, simulation_count)
        gbm.generate_paths()
        stddev = 100 * (np.exp(volatility ** 2) - 1)
        avg = np.average(gbm.paths[:, -1])
        self.assertTrue(100 - stddev < avg < 100 + stddev)

    @staticmethod
    def test_plot():
        initial_asset_value: float = 100
        drift: float = 0.1
        volatility: float = 0.2
        time_step_size: float = 0.01
        time_to_maturity: float = 1
        simulation_count = 1000
        sp = GBM(initial_asset_value, drift, volatility, time_to_maturity, time_step_size, simulation_count)
        sp.generate_paths()
        sp.plot_paths()
