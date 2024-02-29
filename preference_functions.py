import numpy as np
from scipy.stats import beta
from scipy.stats import norm


class PreferenceCurve:
    def __init__(self, lower_bound, mode, upper_bound):
        self.lower_bound = lower_bound
        self.mode = mode
        self.upper_bound = upper_bound
        self.x_values = 0
        self.y_values = 0

    def linear(self):
        # Create x values
        x = np.linspace(self.lower_bound, self.upper_bound, 1000)
        self.x_values = x
        # Create y values based on the mode
        if self.mode == self.lower_bound:
            m = -100 / np.max(x)
            b = 100
            # Calculate y values
            y = m * x + b
        elif self.mode == self.upper_bound:
            m = 100 / np.max(x)
            # Calculate y values
            y = m * x
        else:
            y = np.where(
                x <= self.mode,
                100 * (x - self.lower_bound) / (self.mode - self.lower_bound),
                100 * (self.upper_bound - x) / (self.upper_bound - self.mode),
            )
        self.y_values = y

    def beta_PERT(self):
        x = np.linspace(
            self.mode - self.lower_bound, self.mode + self.upper_bound, 1000
        )
        self.x_values = x
        peak = self.mode  # The location of the peak
        min_val = self.mode - self.lower_bound  # The minimum value
        max_val = self.mode + self.upper_bound  # The maximum value

        # Calculate the shape parameters for the Beta distribution
        alpha = 1 + 4 * (peak - min_val) / (max_val - min_val)
        beta_b = 1 + 4 * (max_val - peak) / (max_val - min_val)

        # Generate the Beta distribution
        y = beta.pdf((x - min_val) / (max_val - min_val), alpha, beta_b)
        self.y_values = (y * (100 / np.max(y))).astype(int)

    def parabolic(self):
        # Create x values
        x = np.linspace(self.lower_bound, self.upper_bound, 1000)
        self.x_values = x

        # Create y values based on the mode
        if self.lower_bound == self.mode:  # rightward facing parabola
            self.y_values = (
                -100 * ((x - self.mode) ** 2) / (self.upper_bound - self.mode) ** 2
                + 100
            )
            self.y_values[x < self.mode] = 0
        elif self.upper_bound == self.mode:  # leftward facing parabola
            self.y_values = (
                -100 * ((x - self.mode) ** 2) / (self.mode - self.lower_bound) ** 2
                + 100
            )
            self.y_values[x > self.mode] = 0
        else:  # full parabola
            # Create y values based on the mode
            self.y_values = (
                -100 * ((x - self.mode) ** 2) / (self.mode - self.lower_bound) ** 2
                + 100
            )

    def normal_distribution(self):
        # Create x values
        x = np.linspace(self.lower_bound, self.upper_bound, 1000)
        self.x_values = x
        # Create y values based on the mode
        std_dev = (
            self.upper_bound - self.mode
        ) / 3  # Assuming 99.7% of data within [lower_bound, upper_bound]
        y = norm.pdf(x, self.mode, std_dev)
        self.y_values = y / np.max(y) * 100  # Normalize to [0, 100]
