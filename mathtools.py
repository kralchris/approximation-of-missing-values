"""
Tools - mathematical tools and decorators for data processing
        and advanced data manipulation


@author: Kristijan <kristijan.sarin@gmail.com>
"""

import numpy as np
from scipy.stats import gmean, hmean, mode, linregress


class Means:
    @staticmethod
    def arithmetic_mean(values):
        return sum(values) / len(values)

    @staticmethod
    def geometric_mean(values):
        return gmean(values)

    @staticmethod
    def harmonic_mean(values):
        return hmean(values)

    @staticmethod
    def quadratic_mean(values):
        return np.sqrt(np.mean(np.square(values)))

    @staticmethod
    def weighted_mean(values, weights):
        return np.average(values, weights=weights)

    @staticmethod
    def trimean(values):
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        return (q1 + 2*q2 + q3) / 4

    @staticmethod
    def power_mean(values, r):
        return np.power(np.mean(np.power(values, r)), 1/r)

    @staticmethod
    def contraharmonic_mean(values):
        return sum(np.square(values)) / sum(values)

    @staticmethod
    def geometric_harmonic_mean(values):
        return np.sqrt(gmean(values) * hmean(values))

    @staticmethod
    def logarithmic_mean(a, b):
        if a == b:
            return a
        else:
            return (b - a) / np.log(b / a)

    @staticmethod
    def median(values):
        return np.median(values)

    @staticmethod
    def mode_value(values):
        return mode(values).mode[0]

    @staticmethod
    def generalized_mean(values, r):
        return np.power(np.mean(np.power(values, r)), 1/r)

    @staticmethod
    def heronian_mean(a, b):
        return (a + b + np.sqrt(a*b)) / 3

    @staticmethod
    def stolarsky_mean(a, b, p):
        if a == b:
            return a
        else:
            return np.power((np.power(b, p+1) - np.power(a, p+1)) / ((p+1) * (b - a)), 1/p)


class DataProcessor:
    @staticmethod
    def apply_mean(mean_func):
        def decorator(func):
            def wrapper(*args, **kwargs):
                data = func(*args, **kwargs)
                for row in data:
                    if None in row:
                        values = [x for x in row if x is not None]
                        mean_value = mean_func(values)
                        row[:] = [mean_value if x is None else x for x in row]
                return data
            return wrapper
        return decorator

    @staticmethod
    def linear_interpolation(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            for row in data:
                # Indices and values for existing data points
                indices = [i for i, x in enumerate(row) if x is not None]
                values = [row[i] for i in indices]
                if len(indices) > 1:  # Interpolation is possible
                    interp_func = np.interp(range(len(row)), indices, values)
                    row[:] = [interp_func[i]
                              if x is None else x for i, x in enumerate(row)]
            return data
        return wrapper

    @staticmethod
    def regression(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            for row in data:
                indices = [i for i, x in enumerate(row) if x is not None]
                values = [row[i] for i in indices]
                if len(indices) > 1:  # Regression is possible
                    slope, intercept, _, _, _ = linregress(indices, values)
                    row[:] = [(slope * i + intercept)
                              if x is None else x for i, x in enumerate(row)]
            return data
        return wrapper
