class HullWhite:
    """
    Notes:
        • According to https://quant.stackexchange.com/questions/10194/simulating-the-short-rate-in-the-hull-white-model
          you don't need option prices except for generalised Hull-White
        • Apparently you can calibrate theta as piece-wise constant.
    """
    def __init__(self, alpha, sigma, initial_curve):
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve
