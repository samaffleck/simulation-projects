"""
Monte carlo simulation for the insertion of sodium into hard carbon.
This code is not to be run on the HEC as it includes plotting using Matplotlib.
"""
import uuid
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
from scipy import integrate


class MonteCarlo:

    def __init__(self):

        # All of the arguments required passed through the command line.
        self.parser = argparse.ArgumentParser(description="Monte carlo simulation for sodation into hard carbon")
        self.parser.add_argument('--sites', type=int, metavar='',
                                 help='Total number of sites including nanopores and interlayers', default=400)
        self.parser.add_argument('--sps', type=int, metavar='',
                                 help='Number of Monte Carlo steps per site', default=500)
        self.parser.add_argument('--l', type=float, metavar='', help='Fraction of total sites in the interlayers',
                                 default=0.329217689)
        self.parser.add_argument('--eps1', type=float, metavar='', help='Point interaction term for interlayers in eV',
                                 default=-0.37289072)
        self.parser.add_argument('--eps2', type=float, metavar='',
                                 help='Point term for nanopores - priori heterogeneity in eV', default=-0.021718645)
        self.parser.add_argument('--g2', type=float, metavar='', help='g2 term', default=-0.046075307)
        self.parser.add_argument('--g3', type=float, metavar='', help='g3 term', default=0.03)
        self.parser.add_argument('--a', type=float, metavar='', help='a term', default=-0.684477326)
        self.parser.add_argument('--b', type=float, metavar='', help='b term', default=1.873973403)
        self.parser.add_argument('--c', type=float, metavar='', help='c term', default=1.686422347)
        self.parser.add_argument('--sample_frequency', type=int, metavar='',
                                 help='Number of mcs before a sample is taken', default=200)
        self.parser.add_argument('--T', type=float, metavar='', help='Temperature', default=288)
        self.parser.add_argument('--eps1_max', type=float, metavar='', help='Maximum point value for interlayers (uniform)',
                                 default=-0.1)
        self.parser.add_argument('--eps1_min', type=float, metavar='', help='Minimum point value for interlayers (uniform)',
                                 default=-1.35)
        self.parser.add_argument('--eps1_mean', type=float, metavar='', help='Mean value for interlayers (norm)',
                                 default=-0.7)
        self.parser.add_argument('--eps1_sig', type=float, metavar='',
                                 help='Standard deviation for the point values for interlayers (norm)', default=0.45)
        self.parser.add_argument('--eps1_low', type=float, metavar='',
                                 help='Most negative interlayer energy (tri)', default=-1.65)
        self.parser.add_argument('--eps1_high', type=float, metavar='',
                                 help='Most positive interlayer energy (tri)', default=0.08)
        self.parser.add_argument('--eps1_power_low', type=float, metavar='',
                                 help='Most negative interlayer energy (power)', default=-3.3)
        self.parser.add_argument('--eps1_power_high', type=float, metavar='',
                                 help='Most positive interlayer energy (power)', default=-0.06)
        self.parser.add_argument('--eps1_power_a', type=float, metavar='',
                                 help='Power constant for power distribution', default=6)
        self.parser.add_argument('--mu_list', type=float, nargs='+', metavar='',
                                 help='List of chemical potentials to loop through',
                                 default=[-1.6, -1.5, -1.4, -1.35, -1.3, -1.25, -1.2, -1.16, -1.12, -1.08, -1.04, -1.0, -0.96, -0.88, -0.80, -0.72, -0.64,
                                          -0.56, -0.5, -0.46, -0.42, -0.38, -0.34, -0.3, -0.26, -0.22, -0.20, -0.18,
                                          -0.16, -0.14, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.10, -0.095,
                                          -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.06, -0.05, -0.04, -0.03,
                                          -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12])
        self.parser.add_argument('--distribution', type=int, metavar='', help='Type of distribution: 0:Uniform, '
                                                                              '1:Normal '
                                                                              '2:Power distribution '
                                                                              '3: Triangular '
                                                                              '4: Exponential expression (No distribution)', default=0)

        self.args = self.parser.parse_args()  # Stores all of the arguments

        self.eps1 = self.args.eps1_mean  # Point term for the interlayers.
        self.eps2 = self.args.eps2  # Constant point value for the nanopores.

        # Constants for the power distribution.
        self.a = self.args.eps1_power_a
        self.minp = self.args.eps1_power_low
        self.maxp = self.args.eps1_power_high

        if self.args.distribution == 0:
            self.args.l = 0.25  # This is due to the nature of the uniform distribution where the optimum L differs.
        if self.args.distribution == 0:
            self.args.l = 0.27  # This is due to the nature of the normal distribution where the optimum L differs.

        self.lattice1_sites = int(self.args.sites * self.args.l)  # Number of sites in the interlayers.
        self.lattice2_sites = self.args.sites - self.lattice1_sites  # Number of sites in the nanopores.

        # Stores the occupation number for all sites in the lattice.
        # self.lattice[0] = interlayers and self.lattice[1] = nanopores
        # Occupation number of 1 is filled and 0 means vacant.
        self.lattice = [np.zeros(self.lattice1_sites), np.zeros(self.lattice2_sites)]

        # Checks what distribution we want to use and initialises randomly the point energies for each lattice.
        if self.args.distribution == 0:
            self.lattice_energy = [np.random.uniform(self.args.eps1_min, self.args.eps1_max, self.lattice1_sites),
                                   np.zeros(
                                       self.lattice2_sites)]  # Stores the point energy for every site in the lattice.
        elif self.args.distribution == 1:
            self.lattice_energy = [np.random.normal(self.args.eps1_mean, self.args.eps1_sig, self.lattice1_sites),
                                   np.zeros(
                                       self.lattice2_sites)]  # Stores the point energy for every site in the lattice.
        elif self.args.distribution == 2:
            self.lattice_energy = [np.random.power(self.a, self.lattice1_sites) * -1 * (self.minp - self.maxp) + self.minp,
                                   np.zeros(
                                       self.lattice2_sites)]  # Stores the point energy for every site in the lattice.
        elif self.args.distribution == 3:
            self.lattice_energy = [np.random.triangular(self.args.eps1_low, self.args.eps1_high, self.args.eps1_high, self.lattice1_sites),
                                   np.zeros(
                                       self.lattice2_sites)]  # Stores the point energy for every site in the lattice.
        elif self.args.distribution == 4:  # No distribution
            self.lattice_energy = [np.zeros(self.lattice1_sites), np.zeros(self.lattice2_sites)]
            for i, site in enumerate(self.lattice_energy[0]):
                self.lattice_energy[0][i] = self.args.eps1

        for i, site in enumerate(self.lattice_energy[1]):
            self.lattice_energy[1][i] = self.eps2

        print(self.lattice_energy)

        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.T = self.args.T  # Temperature in K

    def plot_results(self, results_array, avrN_data, avrU_data):
        """
        Post processing of results and plots all of the parameters of interest.
        """

        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 1

        # Converts the numpy array into a pandas data frame.
        results_df = pd.DataFrame(data=results_array, columns=["Interlayer mole fraction",
                                                               "Nanopore mole fraction",
                                                               "Total mole fraction",
                                                               "Chemical potential",
                                                               "Partial molar entropy",
                                                               "dq/de",
                                                               "Partial molar enthalpy"])

        results_df = results_df.replace(0, pd.np.nan).dropna(axis=0, how='all')  # For the rows with all '0' entries they are replaced with 'nan' and then these rows are dropped.
        results_df = results_df.replace(pd.np.nan, 0)  # As some legitimate 0 entries such as 0 volts we flip back the remaining from 'nan' to 0.

        # Integrates the p.m. entropy
        entropy_list = integrate.cumtrapz(results_df['Partial molar entropy'], results_df['Total mole fraction'],
                                          initial=0)  # Contains the entropy values
        results_df['Entropy'] = entropy_list

        # Only used when we want to record the fluctuations in U and N (Rare).
        avrN_df = pd.DataFrame(data=avrN_data)
        avrU_df = pd.DataFrame(data=avrU_data)

        uid = str(uuid.uuid1())  # Creates a unique code.
        cwd = os.getcwd()  # Gets the current working directory
        path = cwd + "/results"
        Path(path).mkdir(parents=True, exist_ok=True)  # Create a folder if it doesn't already exist.
        results_df.to_csv(path + "/Na_monte_carlo_results_" + uid + ".csv", index=False)
        # avrU_df.to_csv(path + "/average_U.csv")
        # avrN_df.to_csv(path + "/average_N.csv")

        dfE = pd.read_csv(path + "/experimental_data.csv")  # gets experimental data

        # Rescale the x-axis
        ratio_of_capacities = 272.4 / 338.313338
        dfE["x_real"] = ratio_of_capacities * dfE["x"]

        # vertical shift on p.m. entropy for vibrational effect
        vibrational_shift = 0.0149  # eV K
        dfE["Entropy dS/dx"] = dfE["Entropy dS/dx"] - vibrational_shift

        # Rescale voltage profile and p.m. enthalpy
        results_df["adjusted voltage"] = results_df["Chemical potential"] * ratio_of_capacities
        results_df["adjusted enthalpy"] = results_df["Partial molar enthalpy"] * ratio_of_capacities
        results_df["adjusted entropy"] = results_df["Partial molar entropy"] * ratio_of_capacities
        results_df["adjusted dq/de"] = results_df["dq/de"] * (1 / ratio_of_capacities) ** 2

        # Differentiate the p.m. enthalpy to get the second derivative.
        pm_enthalpy = np.array(results_df['adjusted enthalpy'])
        mole_fraction = np.array(results_df['Total mole fraction'])
        secder_enthalpy = np.gradient(pm_enthalpy, mole_fraction)
        results_df['secder enthalpy'] = secder_enthalpy

        # PLOTS THE ANALYTICAL SOLUTION
        points = 1000
        x_pos = np.linspace(0, 1, points)
        y_pos = np.linspace(0, 1, points)
        s_x = np.linspace(0, 1, points)
        s_y = np.linspace(0, 1, points)
        l = 0.3292
        R = -0.0000862  # eV/K.Site
        T = 288  # K
        for index, x in enumerate(x_pos):
            if x < l:
                s_y[index] = (R * (x * np.log(x/l) - (x-l)*np.log((l-x)/l))) * T
                y_pos[index] = T * R * (np.log(x/l) - np.log((l-x)/l))
            else:
                s_y[index] = (R * l * ((x/l - 1) * np.log(x/l - 1) + (1-x)/l * np.log((1-x)/l) - (1-l)/l * np.log((1-l)/l))) * T
                y_pos[index] = T * R * (np.log(x/l - 1) - np.log(1/l - x/l))

        fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

        lw = 0.7  # Line width

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[0, 0], x='x_real', y='OCV')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[0, 0], x='x', y='OCV')
        ax1 = results_df.plot(linestyle='-', color='black', lw=lw, marker='o', markeredgecolor='black',
                              markersize=3, ax=axes[0, 0], x='Total mole fraction', y='adjusted voltage')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[0, 1], x='x_real', y='Entropy dS/dx')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[0, 1], x='x', y='Entropy dS/dx')
        ax2 = results_df.plot(linestyle='-', color='black', lw=lw, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[0, 1], x='Total mole fraction', y='adjusted entropy')

        ax2.plot(x_pos, y_pos, linewidth=lw)  # Plots the ideal p.m. entropy

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[1, 0], x='OCV', y='dQdV')

        ax3 = results_df.plot(linestyle='-', color='blue', lw=lw, marker='o', markeredgecolor='black',
                              markersize=3, ax=axes[1, 0], x='Chemical potential', y='adjusted dq/de')

        dfE.plot(linestyle='-', color='darkgreen', lw=lw, ax=axes[1, 1], x='x_real', y='Enthalpy dH/dx')
        dfE.plot(linestyle='-', color='darkblue', lw=lw, ax=axes[1, 1], x='x', y='Enthalpy dH/dx')
        ax4 = results_df.plot(linestyle='-', color='black', lw=lw, marker='o', markeredgecolor='black',
                              markersize=3, ax=axes[1, 1], x='Total mole fraction', y='adjusted enthalpy')

        ax5 = results_df.plot(linestyle='-', color='black', lw=lw, marker='o', markeredgecolor='black',
                              markersize=3, ax=axes[2, 1], x='Total mole fraction', y='secder enthalpy')

        ax6 = results_df.plot(linestyle='-', color='black', lw=lw, marker='o', markeredgecolor='green',
                              markersize=3, ax=axes[2, 0], x='Total mole fraction', y='Entropy')
        ax6.plot(s_x, s_y, linewidth=lw)  # Plots the entropy.

        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
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
        ax2.set_ylabel('dS/dx $[eV K/site]$')
        ax3.set_ylabel('dq/de [$\mathregular{eV^{-1}}$]')
        ax4.set_ylabel('$dH/dx$ $[eV/site]$')
        ax5.set_ylabel('$d^2H/dx^2$ $[eV/site]$')
        ax6.set_ylabel('S $[eV K/site]$')

        if self.args.distribution == 0:
            plt.suptitle(
                "Distribution: Uniform | Sites: {} | Steps per site: {} | L: {} | eps1_min: {} | eps1_max: {} | eps2: {}".format(
                    self.args.sites, self.args.sps, self.args.l, self.args.eps1_min, self.args.eps1_max, self.args.eps2))
        elif self.args.distribution == 1:
            plt.suptitle(
                "Distribution: Normal | Sites: {} | Steps per site: {} | L: {} | eps1_mean: {} | eps1_sig: {} | eps2: {}".format(
                    self.args.sites, self.args.sps, self.args.l, self.args.eps1_mean, self.args.eps1_sig, self.args.eps2))
        elif self.args.distribution == 3:
            plt.suptitle(
                "Distribution: Triangle | Sites: {} | Steps per site: {} | L: {} | eps1_low: {} | eps1_high: {} | eps2: {}".format(
                    self.args.sites, self.args.sps, self.args.l, self.args.eps1_low, self.args.eps1_high, self.args.eps2))
        elif self.args.distribution == 4:
            plt.suptitle(
                "Exponential equation | Sites: {} | Steps per site: {} | L: {} | eps1: {} | eps2: {} | a: {} | b: {} | c: {} | g2: {}  | g3: {}   ".format(
                    self.args.sites, self.args.sps, self.args.l, self.args.eps1, self.args.eps2, self.args.a, self.args.b, self.args.c, self.args.g2, self.args.g3))

        parameter_file = open(path + "/Input_arguments_" + uid + ".txt", "w")
        parameter_file.write(str(self.args))
        parameter_file.close()

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        # fig_path = cwd + "/Na_plot_results.png"
        # plt.savefig(path + "/Na_monte_carlo_plot_" + uid + ".png")
        plt.show()

    def calculate_u(self):
        """
        Calculates the Internal energy of the whole lattice.
        """
        if self.args.distribution == 4:
            N1 = np.sum(self.lattice[0])  # Number of filled interlayer sites
            N2 = np.sum(self.lattice[1])  # Number of filled nanopores
            n1 = N1/self.lattice1_sites  # Mole fraction of interlayers
            n2 = N2/self.lattice2_sites  # Mole fraction of nanopores
            M2 = self.lattice2_sites

            nano_term = (self.eps2 * n2 * M2) + (self.args.g2 * M2 * (n2 ** 2)) + (self.args.g3 * M2 * (n2 ** 3))
            eps1 = self.args.eps1 + self.args.a * math.exp(-self.args.b*n1**self.args.c)
            return eps1 * N1 + nano_term
        else:
            N2 = np.sum(self.lattice[1])  # Number of filled nanopores
            n2 = N2 / self.lattice2_sites  # Mole fraction of nanopores
            M2 = self.lattice2_sites
            nano_term = (self.eps2 * n2 * M2) + (self.args.g2 * M2 * (n2 ** 2)) + (self.args.g3 * M2 * (n2 ** 3))

            return np.sum(np.multiply(self.lattice[0], self.lattice_energy[0])) + nano_term

    def site_h(self, mu, lattice_index, random_index):
        if self.args.distribution == 4:
            if lattice_index == 0:  # interlayers
                N1 = np.sum(self.lattice[0])  # Number of filled interlayer sites
                n1 = N1/self.lattice1_sites  # Mole fraction of interlayers
                eps1 = self.args.eps1 + self.args.a * math.exp(-self.args.b*n1**self.args.c)
                return eps1 * N1 - self.lattice[lattice_index][random_index] * mu
            else:
                n2 = np.sum(self.lattice[1])/self.lattice2_sites  # Mole fraction of nanopores
                M2 = self.lattice2_sites
                nano_term = n2 * ((self.eps2 * M2) + (self.args.g2 * M2 * n2) + (self.args.g3 * M2 * (n2 * n2)))
                return nano_term - self.lattice[lattice_index][random_index] * mu
        else:
            if lattice_index == 0:  # interlayers
                return self.lattice[lattice_index][random_index] * self.lattice_energy[lattice_index][random_index] - \
                       self.lattice[lattice_index][random_index] * mu
            else:  # Pores
                n2 = np.sum(self.lattice[1]) / self.lattice2_sites  # Mole fraction of nanopores
                M2 = self.lattice2_sites
                nano_term = (self.eps2 * M2 * n2) + (self.args.g2 * M2 * n2 * n2) + (self.args.g3 * M2 * (n2 * n2 * n2))
                return nano_term - self.lattice[lattice_index][random_index] * mu

    def monte_carlo(self, mu):
        """
        Main function - metropolis monte carlo algorithm
        """

        rand_l = random.random()  # Random number between 0 and 1
        if rand_l < self.args.l:
            lattice_index = 0
        else:
            lattice_index = 1

        random_index = random.randint(0, np.size(
            self.lattice[lattice_index]) - 1)  # Selects a site depending which layer we are in.

        current_h = self.site_h(mu, lattice_index, random_index)

        # Perform a swap
        if self.lattice[lattice_index][random_index] == 0:
            self.lattice[lattice_index][random_index] = 1
        else:
            self.lattice[lattice_index][random_index] = 0

        new_h = self.site_h(mu, lattice_index, random_index)
        delta_h = new_h - current_h

        if delta_h > 0:
            rand_p = random.random()
            p = math.exp(-delta_h / (self.kb * self.T))
            if rand_p > p:
                # Perform a swap - Overall we didn't swap the occupation number for this case.
                if self.lattice[lattice_index][random_index] == 0:
                    self.lattice[lattice_index][random_index] = 1
                else:
                    self.lattice[lattice_index][random_index] = 0

    def run_simulation(self):
        rows = len(self.args.mu_list) - 1
        row_count = 0
        results_array = np.zeros((rows + 1, 7))  # Columns for delta S, x, mu, enthalpy, etc.

        steps = self.args.sps * self.args.sites
        average_rows = steps / self.args.sample_frequency
        data_array = np.zeros((6, int(average_rows)))  # Records the parameters at the sample frequency

        avrU_array = np.zeros(
            (len(self.args.mu_list), int(average_rows)))  # Records averaging decay of U for all chemical potentials.
        avrN_array = np.zeros(
            (len(self.args.mu_list), int(average_rows)))  # Records averaging decay of n for all chemical potentials.
        monitor_averaging = False  # True if you want to save the U and N during equilibration (Default is False).

        for count, mu in enumerate(self.args.mu_list):
            print("Chemical potential:", mu)

            # Runs equilibration steps
            avr_count = 0
            for i in range(steps):
                self.monte_carlo(mu)
                if i % self.args.sample_frequency == 0 and monitor_averaging == True:
                    u = self.calculate_u()
                    n = np.sum(self.lattice[0]) + np.sum(self.lattice[1])
                    avrU_array[count][avr_count] = u
                    avrN_array[count][avr_count] = n
                    avr_count += 1

            # Runs the averaging steps.
            data_count = 0
            for i in range(steps):
                self.monte_carlo(mu)
                if i % self.args.sample_frequency == 0:
                    u = self.calculate_u()
                    n = np.sum(self.lattice[0]) + np.sum(self.lattice[1])
                    data_array[0][data_count] = u  # Internal energy
                    data_array[1][data_count] = n  # Occupancy
                    data_array[2][data_count] = u * n  # Occupancy * Internal energy
                    data_array[3][data_count] = n * n  # Occupancy ** 2
                    data_array[4][data_count] = np.sum(self.lattice[0])
                    data_array[5][data_count] = np.sum(self.lattice[1])
                    data_count += 1

            mean_u = np.average(data_array[0])
            mean_n = np.average(data_array[1])
            mean_un = np.average(data_array[2])
            mean_nn = np.average(data_array[3])
            mean_x1 = np.average(data_array[4]) / self.lattice1_sites
            mean_x2 = np.average(data_array[5]) / self.lattice2_sites
            mean_x = mean_n / self.args.sites
            variance = mean_nn - mean_n ** 2
            cov_un = mean_un - (mean_u * mean_n)

            print('Interlayer mole fraction:', mean_x1, 'Nanopore mole fraction:', mean_x2, 'Total mole fraction:',
                  mean_x)
            print("--------")

            if variance != 0:
                delta_entropy = (cov_un / variance) - mu
                results_array[row_count, 0] = mean_x1
                results_array[row_count, 1] = mean_x2
                results_array[row_count, 2] = mean_x
                results_array[row_count, 3] = mu * -1
                results_array[row_count, 4] = delta_entropy  # Partial molar entropy
                results_array[row_count, 5] = variance / (self.kb * self.T * self.args.sites)  # dq/de
                results_array[row_count, 6] = cov_un / variance  # Partial molar enthalpy
                row_count += 1

        self.plot_results(results_array, avrN_array, avrU_array)


if __name__ == '__main__':
    mc = MonteCarlo()
    mc.run_simulation()
