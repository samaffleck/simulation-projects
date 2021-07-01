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
        """
        This is the main function to plot the results.
        """

        cwd = os.getcwd()  # Gets current working directory.
        path = cwd + "/results"  # This is the folder all the results are stored in.
        dataframes = ["/Na_monte_carlo_results_a_b2655a2f-da59-11eb-b85f-505bc2f6ccb0.csv"]  # This is a list so you can pass multiple csv files to be overlayed on the same plot.
        colours = ['black', 'darkred', 'darkmagenta', 'darkturquoise', 'saddelbrown']  # Array of colours for the lines.

        dfE = pd.read_csv(path + "/experimental_data.csv")  # Reads in the experimental data as a pandas dataframe.

        # Rescale the x-axis of the experimental data.
        ratio_of_capacities = 272.4 / 338.313338  # experimental maximum capacity / theoretical maximum capacity
        dfE["x_theo"] = ratio_of_capacities * dfE["x"]
        # 'x' is the experimental x and 'x_theo' is the theoretical x.

        # Second derivative of enthalpy for experimental data. One w/ respect to the experimental x and one w/ respect to theoretical x.
        secder_enthalpy_experimental_x = np.gradient(np.array(dfE['Enthalpy dH/dx']), np.array(dfE['x']))
        secder_enthalpy_experimental_x_theo = np.gradient(np.array(dfE['Enthalpy dH/dx']), np.array(dfE['x_theo']))
        dfE['secder enthalpy x'] = secder_enthalpy_experimental_x
        dfE['secder enthalpy x theo'] = secder_enthalpy_experimental_x_theo

        # vertical shift on p.m. entropy for vibrational effect
        vibrational_shift = 0.0108  # eV K  this includes being multiplied by the ratio of capacities.
        dfE["Entropy dS/dx"] = (dfE["Entropy dS/dx"]) - vibrational_shift

        # Integrates the p.m. entropy
        entropy_list_experimental = integrate.cumtrapz(dfE['Entropy dS/dx'], dfE['x'],
                                                       initial=0)  # Contains the entropy values
        dfE['Entropy'] = entropy_list_experimental

        dfE['x_new'] = ((dfE['x_theo'] - dfE['x_theo'].iloc[0]) * dfE['x_theo'][73]) / (dfE['x_theo'][73] - dfE['x_theo'].iloc[0])  # Rescales the line so that the experimental data starts at 0.
        dfE['x'] = ((dfE['x'] - dfE['x'].iloc[0]) * dfE['x'][73]) / (dfE['x'][73] - dfE['x'].iloc[0])  # Same as above but for experimental x axis.

        # Calculates the analytical solution
        points = 1000
        x_pos = np.linspace(0, 1, points)  # x for p.m. entropy
        y_pos = np.linspace(0, 1, points)  # y for p.m. etropy
        s_x = np.linspace(0, 1, points)  # x for entropy
        s_y = np.linspace(0, 1, points)  # y for entropy
        l = 0.329217689  # This must be the same as what was used in the main script
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

        #  Calculates the single solid state entropy
        x_ent = np.linspace(0, 1, points)
        y_ent = np.linspace(0, 1, points)
        for index, x in enumerate(x_ent):
            y_ent[index] = T * R * (x * np.log(x) + (1-x) * np.log(1-x))

        # Create plot and formats
        fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
        plt.rc('legend', fontsize=7)
        lw = 0.7  # Line width

        # Plots all of the experimental data
        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[0, 0], x='x_new', y='OCV')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[0, 0], x='x', y='OCV')

        ax2 = dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[0, 1], x='x_new', y='Entropy dS/dx')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[0, 1], x='x', y='Entropy dS/dx')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[1, 0], x='OCV', y='dQdV')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[1, 1], x='x_new', y='Enthalpy dH/dx')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[1, 1], x='x', y='Enthalpy dH/dx')

        ax5 = dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[2, 0], x='x_new', y='Entropy')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[2, 1], x='x_new', y='secder enthalpy x theo')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[2, 1], x='x', y='secder enthalpy x')

        # Iterate through all the data to be plotted
        for count, df in enumerate(dataframes):
            df1 = pd.read_csv(path + df)  # reads file into a dataframe.

            df1 = df1.replace(0, pd.np.nan).dropna(axis=0, how='all')  # For the rows with all '0' entries they are replaced with 'nan' and then these rows are dropped.
            df1 = df1.replace(pd.np.nan, 0)  # As some legitimate 0 entries such as 0 volts we flip back the remaining from 'nan' to 0.

            # Integrates the p.m. entropy
            entropy_list = integrate.cumtrapz(df1['Partial molar entropy'], df1['Total mole fraction'],
                                              initial=0)  # Contains the entropy values
            df1['Entropy'] = entropy_list

            # Rescale voltage profile and p.m. enthalpy by the chain rule.
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

            ax6.plot(s_x, s_y, linewidth=lw, color='red')  # Plots the entropy for l=0.32...
            ax6.plot(x_ent, y_ent, linewidth=lw, color='grey')  # Plots the entropy for solid state solution.

            ax1.set_xlim([0, 1])
            ax2.set_xlim([0, 1])
            ax3.set_xlim([-0.1, 1])
            ax4.set_xlim([0, 1])
            ax5.set_xlim([0, 1])
            ax6.set_xlim([0, 1])

            ax5.set_ylim([0, 6])

            ax1.set_xlabel('Na content $[x]$')
            ax2.set_xlabel('Na content $[x]$')
            ax3.set_xlabel('Voltage $[V]$')
            ax4.set_xlabel('Na content $[x]$')
            ax5.set_xlabel('Na content $[x]$')
            ax6.set_xlabel('Na content $[x]$')

            ax1.set_ylabel('Voltage $[V]$')
            ax2.set_ylabel('dS/dx $[eV K/site]$')
            ax3.set_ylabel('dq/de [$\mathregular{eV^{-1}}$]')
            ax4.set_ylabel('$dH/dx$ $[eV/site]$')
            ax5.set_ylabel('$d^2H/dx^2$ $[eV/site]$')
            ax6.set_ylabel('S $[eV K/site]$')

            ax1.legend(['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data'])
            ax2.legend(
                ['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data',  'Analytical solution'])
            ax3.legend(['Experimental data', 'Monte Carlo data'])
            ax4.legend(['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data'])
            ax5.legend(['Experimental data (Adjusted x)', 'Raw experimental data', 'Monte Carlo data'])
            ax6.legend(['Experimental data', 'Monte Carlo data', 'Analytical solution', 'Solid state solution'], loc='upper right', bbox_to_anchor=(0.75, 0.5))

        plt.show()

    def display_averaging(self):
        """
        This can be used to display the averages by passing through the two average files for U and N. To get these files run the code with monitor_averaging = True.
        """

        cwd = os.getcwd()
        path = cwd + "/results"
        df1 = pd.read_csv(path + "/average_U.csv")  # black line
        df2 = pd.read_csv(path + "/average_N.csv")  # green line
        chem = 25  # from 0 to 35

        s1 = df1.iloc[chem]
        s1.plot()

        plt.show()

    def test_uniform(self):
        """
        Shows the pdf for a uniform distribution
        """

        s = np.random.uniform(-1.35, 0.5, 5000)
        plt.hist(s, 30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()

    def test_normal(self):
        """
        Shows the pdf for a normal distribution
        """
        s = np.random.normal(-0.42, 0.55, 5000)
        plt.hist(s, 30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()

    def test_triangular(self):
        """
        Shows the pdf for a triangular distribution
        """
        s = np.random.triangular(-1.65, 0.08, 0.08, 5000)
        plt.hist(s, bins=30, density=False)
        plt.xlabel('Interlayer point energy [eV]')
        plt.ylabel('Frequency')
        plt.show()

    def test_power(self):
        """
        Shows the pdf for a power distribution
        """
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
