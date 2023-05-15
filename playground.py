import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logarithmic_function(x, a, b):
    return a * np.log(x) + b


def linear_function(x, a, b):
    return a * x + b


def fit_logarithm(x_values, y_values):
    params, cov = curve_fit(logarithmic_function, x_values, y_values)
    a, b = params
    uncertainty = np.sqrt(np.diag(cov))

    # Plotting the measured values and logarithmic function
    plt.scatter(x_values, y_values, label='Naměřené hodnoty')
    label = f"$ f(x) = a \cdot \ln(x) + b$, kde \n " \
            f"$a = {a:.3f} \pm {uncertainty[0]:.3f}$, \n " \
            f"$b = {b:.3f} \pm {uncertainty[1]:.3f}$"
    plt.plot(x_values, logarithmic_function(x_values, a, b), color='r', label=label)

    plt.xlabel('Ztrátový odpor [MΩ]')
    plt.ylabel('Stejnosměrné výstupní napětí [V]')
    plt.legend()
    plt.grid()
    plt.show()

    return params, uncertainty


def fit_linear(x_values, y_values):
    params, cov = curve_fit(linear_function, x_values, y_values)
    a, b = params
    uncertainty = np.sqrt(np.diag(cov))

    # Plotting the measured values and linear function
    plt.scatter(x_values, y_values, label='Naměřené hodnoty')
    label = f"$ f(x) = a \cdot x + b$, kde \n " \
            f"$a = {a:.3f} \pm {uncertainty[0]:.3f}$, \n " \
            f"$b = {b:.3f} \pm {uncertainty[1]:.3f}$"
    plt.plot(x_values, linear_function(x_values, a, b), color='r', label=label)

    plt.xlabel('Kapacita kondenzátoru [pF]')
    plt.ylabel('Stejnosměrné výstupní napětí [V]')
    plt.legend()
    plt.grid()
    plt.show()

    return params, uncertainty


# 2.025
# 3.968
# difference = 1.943
# 1/1.943 = 0.515

# 1.043
# 2.043

def main():
    # Resistance (C=150pF)
    resistance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    voltage = np.array([3.002, 2.634, 2.554, 2.522, 2.507, 2.497, 2.492, 2.488, 2.485, 2.482])

    # Fit the logarithmic function
    params, uncertainty = fit_logarithm(resistance, voltage)

    # Capacitance
    capacitance = np.array([90, 120, 150, 180, 210, 240, 270, 300])
    voltage = np.array([1.929, 2.217, 2.507, 2.799, 3.089, 3.381, 3.674, 3.968])
    params, uncertainty = fit_linear(capacitance, voltage)


if __name__ == '__main__':
    main()
