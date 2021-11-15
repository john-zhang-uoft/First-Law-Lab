import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp


def get_fan_power(file):
    # Load data
    data = pd.read_csv(file, sep='\t', skiprows=2)

    # Set constants
    P1 = ufloat(0.001, 0.00005) * 735.499    # Lab power output, [W]
    P1 = np.ones(data.shape[0]) * P1

    n1 = ufloat(4200, 0.5)  # Lab shaft speed, [rpm]
    n1 = np.ones(data.shape[0]) * n1

    n2 = ufloat(2000, 0.5)  # Operating shaft speed, [rpm]
    n2 = np.ones(data.shape[0]) * n2

    rho1 = ufloat(1.225, 0.0005)  # Lab air density, [kg/m^3]
    rho1 = np.ones(data.shape[0]) * rho1

    R_air = ufloat(287.05, 0.005)  # Air gas constant, [J/kg*K]
    R_air = np.ones(data.shape[0]) * R_air

    # Create new column for operating fan power

    data['Operating Power'] = 0

    # First convert to pascals and add gauge pressure (PSI) to atmospheric pressure (Hg)
    Pressure = (unp.uarray(data['P1(PSI)'].to_numpy(), np.ones(data.shape[0]) * 0.05) * 6894.757) + (
            ufloat(29.50, 0.005) * 3386.389)

    # Convert Celsius to Kelvin
    Temperature = unp.uarray(data['T1(Deg C)'].to_numpy(), np.ones(data.shape[0]) * 0.05) + 273.15

    return P1 * (n2 / n1) ** 3 * Pressure / (R_air * Temperature * rho1)


if __name__ == '__main__':

    files = ['data/part_1a.txt', 'data/part_1b.txt', 'data/part_1c.txt', 'data/part_1d.txt',
             'data/part_2a.txt', 'data/part_2b.txt', 'data/part_2c.txt', 'data/part_2d.txt']

    results = {}

    # plot

    for i in range(len(files)):
        time = pd.read_csv(files[i], sep='\t', skiprows=2)
        time = time['Time(s)']

        data = get_fan_power(files[i])
        results[f'{files[i][5:12]}'] = np.trapz(x=time, y=data)
        plt.scatter(time, unp.nominal_values(data), s=3)
        plt.title(f'Propeller Power vs Time for P{files[i][6:12]}')
        plt.xlabel('Time(s)')
        plt.ylabel('Power(W)')
        fig = plt.gcf()

        plt.show()
        fig.savefig(f'P{files[i][6:12]}.png')

    print(results)
