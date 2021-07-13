import matplotlib.pyplot as plt
import numpy as np


class SurvivalCurve:
    def __init__(self, **kwargs): #, tenors, probabilities_of_default, recovery_rate):
        self.tenors = kwargs.pop('tenors', None)
        self.probabilities_of_default = kwargs.pop('probabilities_of_default', None)
        self.recovery_rate = kwargs.pop('recovery_rate', None)
        self.survival_probabilities = kwargs.pop('survival_probabilities', None)
        if self.survival_probabilities is None:
            self.survival_probabilities = 1 - self.probabilities_of_default

        if not self.tenors.__contains__(0.0):
            self.tenors = np.append(0, self.tenors)
            self.probabilities_of_default = np.append(0, self.probabilities_of_default)


    def get_probabilities_of_default(self, interpolation_tenors):
        # linear interpolation with flat extrapolation
        # y = (y[i+1] - y[i])/(x[i+1] - x[i]) * (x - x[i]) + y[i]
        return np.interp(interpolation_tenors, self.tenors, self.probabilities_of_default)

    def get_survival_probabilities(self, interpolation_tenors):
        # linear interpolation with flat extrapolation
        # y = (y[i+1] - y[i])/(x[i+1] - x[i]) * (x - x[i]) + y[i]
        return np.interp(interpolation_tenors, self.tenors, self.survival_probabilities)

    def get_hazard_rate(self, interpolation_tenors):
        pd = self.get_survival_probabilities(interpolation_tenors)
        return -np.log(pd) / interpolation_tenors

    def plot_curve(self, values_to_plot='probabilities_of_default'):
        if values_to_plot.lower() == 'probabilities_of_default':
            plt.plot(self.tenors, self.probabilities_of_default)
        plt.show()

