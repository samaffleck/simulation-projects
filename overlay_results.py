"""
This code plots the csv results from the hard carbon monte carlo simulations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate


class DisplayResults:

    def __init__(self):
        pass

    def display(self):
        cwd = os.getcwd()
        path = cwd + "/results"
        dataframes = ["/Na_monte_carlo_results_91eff520-d39e-11eb-bb5f-505bc2f6ccb0.csv"]
        colours = ['black', 'darkred', 'darkmagenta', 'darkturquoise', 'saddelbrown']

        dfE = pd.read_csv(path + "/experimental_data.csv")  # grey line

        # Rescale the x-axis of the experimental data.
        ratio_of_capacities = 272.4 / 338.313338
        dfE["x_real"] = ratio_of_capacities * dfE["x"]

        # Second derivative of enthalpy for experimental data.
        secder_enthalpy_experimental_x = np.gradient(np.array(dfE['Enthalpy dH/dx']), np.array(dfE['x']))
        secder_enthalpy_experimental_xreal = np.gradient(np.array(dfE['Enthalpy dH/dx']), np.array(dfE['x_real']))
        dfE['secder enthalpy x'] = secder_enthalpy_experimental_x
        dfE['secder enthalpy xreal'] = secder_enthalpy_experimental_xreal

        # vertical shift on p.m. entropy for vibrational effect
        vibrational_shift = 0.0149  # eV K
        dfE["Entropy dS/dx"] = dfE["Entropy dS/dx"] - vibrational_shift

        # Integrates the p.m. entropy
        entropy_list_experimental = integrate.cumtrapz(dfE['Entropy dS/dx'], dfE['x_real'],
                                                       initial=0)  # Contains the entropy values
        dfE['Entropy'] = entropy_list_experimental

        # Calculates the analytical solution
        points = 1000
        x_pos = np.linspace(0, 1, points)
        y_pos = np.linspace(0, 1, points)
        s_x = np.linspace(0, 1, points)
        s_y = np.linspace(0, 1, points)
        l = 0.329217689
        R = -0.0000862  # eV/K.Site
        T = 288  # K
        for index, x in enumerate(x_pos):
            if x < l:
                s_y[index] = (R * (x * np.log(x / l) - (x - l) * np.log((l - x) / l))) * T
                y_pos[index] = T * R * (np.log(x / l) - np.log((l - x) / l))
            else:
                s_y[index] = (R * l * (
                            (x / l - 1) * np.log(x / l - 1) + (1 - x) / l * np.log((1 - x) / l) - (1 - l) / l * np.log(
                        (1 - l) / l))) * T
                y_pos[index] = T * R * (np.log(x / l - 1) - np.log(1 / l - x / l))

        # Create plot and formats
        fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
        plt.rc('legend', fontsize=8)
        lw = 0.7  # Line width

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[0, 0], x='x_real', y='OCV')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[0, 0], x='x', y='OCV')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[0, 1], x='x_real', y='Entropy dS/dx')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[0, 1], x='x', y='Entropy dS/dx')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[1, 0], x='OCV', y='dQdV')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[1, 1], x='x_real', y='Enthalpy dH/dx')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[1, 1], x='x', y='Enthalpy dH/dx')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[2, 0], x='x_real', y='Entropy')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[2, 1], x='x_real', y='secder enthalpy xreal')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[2, 1], x='x', y='secder enthalpy x')

        # Iterate through all the data to be plotted
        for count, df in enumerate(dataframes):
            df1 = pd.read_csv(path + df)  # black line

            # Integrates the p.m. entropy
            entropy_list = integrate.cumtrapz(df1['Partial molar entropy'], df1['Total mole fraction'],
                                              initial=0)  # Contains the entropy values
            df1['Entropy'] = entropy_list

            # Rescale voltage profile and p.m. enthalpy
            df1["adjusted voltage"] = df1["Chemical potential"] * ratio_of_capacities
            df1["adjusted enthalpy"] = df1["Partial molar enthalpy"] * ratio_of_capacities
            df1["adjusted entropy"] = df1["Partial molar entropy"] * ratio_of_capacities
            df1["adjusted dq/de"] = df1["dq/de"] * (1/ratio_of_capacities)**2

            # Differentiate the p.m. enthalpy to get the second derivative.
            pm_enthalpy = np.array(df1['adjusted enthalpy'])
            mole_fraction = np.array(df1['Total mole fraction'])
            secder_enthalpy = np.gradient(pm_enthalpy, mole_fraction)
            df1['secder enthalpy'] = secder_enthalpy

            ax1 = df1.plot(linestyle='-', color=colours[count], lw=lw, marker='o', markeredgecolor=colours[count],
                           markersize=2, ax=axes[0, 0], x='Total mole fraction', y='adjusted voltage')

            ax2 = df1.plot(linestyle='-', color=colours[count], lw=lw, marker='o', markeredgecolor=colours[count],
                           markersize=2, ax=axes[0, 1], x='Total mole fraction', y='adjusted entropy')

            ax2.plot(x_pos, y_pos, linewidth=lw, color='red')  # Plots the ideal p.m. entropy

            ax3 = df1.plot(linestyle='-', color=colours[count], lw=lw, marker='o', markeredgecolor=colours[count],
                           markersize=2, ax=axes[1, 0], x='Chemical potential', y='adjusted dq/de')

            ax4 = df1.plot(linestyle='-', color=colours[count], lw=lw, marker='o', markeredgecolor=colours[count],
                           markersize=2, ax=axes[1, 1], x='Total mole fraction', y='adjusted enthalpy')

            ax5 = df1.plot(linestyle='-', color=colours[count], lw=lw, marker='o', markeredgecolor=colours[count],
                           markersize=3, ax=axes[2, 1], x='Total mole fraction', y='secder enthalpy')

            ax6 = df1.plot(linestyle='-', color=colours[count], lw=lw, marker='o', markeredgecolor=colours[count],
                           markersize=3, ax=axes[2, 0], x='Total mole fraction', y='Entropy')

            ax6.plot(s_x, s_y, linewidth=lw, color='red')  # Plots the entropy.

            ax1.set_xlim([0, 1])
            ax2.set_xlim([0, 1])
            ax3.set_xlim([0, 1])
            ax4.set_xlim([0, 1])
            ax5.set_xlim([0, 1])
            ax6.set_xlim([0, 1])

            ax1.set_xlabel('Na content $[x]$')
            ax2.set_xlabel('Na content $[x]$')
            ax3.set_xlabel('Voltage $[V]$')
            ax4.set_xlabel('Na content $[x]$')
            ax5.set_xlabel('Na content $[x]$')
            ax6.set_xlabel('Na content $[x]$')

            ax1.set_ylabel('Voltage $[V]$')
            ax2.set_ylabel('dS/dx $[eV/site]$')
            ax3.set_ylabel('dq/de $\mathregular{eV^{-1}}$')
            ax4.set_ylabel('$dH/dx$ $[eV/site]$')
            ax5.set_ylabel('$d^2H/dx^2$ $[eV/site]$')
            ax6.set_ylabel('S $[eV/site]$')

            # fig.suptitle('')
            ax1.legend(['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data'])
            ax2.legend(
                ['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data',  'Analytical solution'])
            ax3.legend(['Experimental data', 'Monte Carlo data'])
            ax4.legend(['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data'])
            ax5.legend(['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data'])
            ax6.legend(['Experimental data', 'Monte Carlo data', 'Analytical solution'])

        plt.show()

    def display_averaging(self):
        cwd = os.getcwd()
        path = cwd + "/results"
        df1 = pd.read_csv(path + "/average_U.csv")  # black line
        df2 = pd.read_csv(path + "/average_N.csv")  # green line
        chem = 15  # from 0 to 35

        s1 = df1.iloc[chem]
        s1.plot()

        plt.show()

    def test_uniform(self):
        s = np.random.uniform(-1.35, 0.5, 5000)
        plt.hist(s, 30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()

    def test_normal(self):
        s = np.random.normal(-0.42, 0.55, 5000)
        plt.hist(s, 30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()

    def test_triangular(self):
        s = np.random.triangular(-1.65, 0.08, 0.08, 5000)
        plt.hist(s, bins=30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()

    def test_power(self):
        a = 6  # shape
        samples = 5000
        max = -0.06
        min = -3.3
        s = np.random.power(a, samples) * -1 * (min - max) + min
        plt.hist(s, bins=30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()


if __name__ == '__main__':
    dr = DisplayResults()
    dr.display()

    #dr.display_averaging()

    #dr.test_uniform()
    #dr.test_triangular()
    #dr.test_power()
    #dr.test_normal()
