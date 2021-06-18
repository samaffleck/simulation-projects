"""
Monte carlo simulation for the insertion of sodium into hard carbon.
This programme outputs a csv data file that can be run on overlay_results.py.
This code is suitable for the HEC.
"""
import uuid
import numpy as np
import random
import math
import pandas as pd
import os
import argparse
from pathlib import Path

class MonteCarlo:

    def __init__(self):
        # All arguments are optional and can be passed through the command line.
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
                                 help='Point term for nanopores in eV', default=-0.021718645)
        self.parser.add_argument('--g2', type=float, metavar='', help='g2 term', default=-0.046075307)
        self.parser.add_argument('--g3', type=float, metavar='', help='g3 term', default=0.032394847)
        self.parser.add_argument('--a', type=float, metavar='', help='a term', default=-0.684477326)
        self.parser.add_argument('--b', type=float, metavar='', help='b term', default=1.873973403)
        self.parser.add_argument('--c', type=float, metavar='', help='c term', default=1.686422347)
        self.parser.add_argument('--sample_frequency', type=int, metavar='',
                                 help='Number of mcs before a sample is taken', default=200)
        self.parser.add_argument('--T', type=float, metavar='', help='Temperature', default=288)
        self.parser.add_argument('--eps1_max', type=float, metavar='', help='Maximum point value for interlayers (uniform)',
                                 default=-0.12)
        self.parser.add_argument('--eps1_min', type=float, metavar='', help='Minimum point value for interlayers (uniform)',
                                 default=-1.05)
        self.parser.add_argument('--eps1_mean', type=float, metavar='', help='Mean value for interlayers (norm)',
                                 default=-0.352)
        self.parser.add_argument('--eps1_sig', type=float, metavar='',
                                 help='Standard deviation for the point values for interlayers (norm)', default=0.3)
        self.parser.add_argument('--eps1_low', type=float, metavar='',
                                 help='Most negative interlayer energy (tri)', default=-1.65)
        self.parser.add_argument('--eps1_high', type=float, metavar='',
                                 help='Most positive interlayer energy (tri)', default=0.08)
        self.parser.add_argument('--mu_list', type=float, nargs='+', metavar='',
                                 help='List of chemical potentials to loop through',
                                 default=[-1.6, -1.5, -1.4, -1.35, -1.3, -1.25, -1.2, -1.16, -1.12, -1.08, -1.04, -1.0,
                                          -0.96, -0.88, -0.80, -0.72, -0.64, -0.56, -0.5, -0.46, -0.42, -0.38, -0.34,
                                          -0.3, -0.26, -0.22, -0.20, -0.18,-0.16, -0.14, -0.13, -0.125, -0.12, -0.115,
                                          -0.11, -0.105, -0.10, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065,
                                          -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.06,
                                          0.08, 0.1, 0.12])
        self.parser.add_argument('--distribution', type=int, metavar='',
                                 help='Type of distribution: '
                                      '0: Uniform distribution, '
                                      '1: Normal distribution'
                                      '2: Power distribution '
                                      '3: Triangular distribution '
                                      '4: Exponential expression (No distribution)', default=0)

        self.args = self.parser.parse_args()  # Stores all input parameters

        self.eps1 = self.args.eps1  # Point term for interlayers
        self.eps2 = self.args.eps2  # Point term for pores

        # Constants for the power distribution.
        self.a = 10
        self.minp = -3
        self.maxp = -0.377

        self.lattice1_sites = int(self.args.sites * self.args.l)  # Number of sites in the interlayers.
        self.lattice2_sites = self.args.sites - self.lattice1_sites  # Number of sites in the nanopores.

        # Stores the occupation number of all the sites in the lattice.
        # self.lattice[0]: interlayers and self.lattice[1]: nanopores.
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
            self.lattice_energy = [np.zeros(self.lattice1_sites), np.zeros(self.lattice2_sites)]  # Stores the point energy for every site in the lattice.
            for i, site in enumerate(self.lattice_energy[0]):
                self.lattice_energy[0][i] = self.args.eps1

        # Asigns constant point term for all nanopore sites.
        for i, site in enumerate(self.lattice_energy[1]):
            self.lattice_energy[1][i] = self.eps2

        print(self.lattice_energy)

        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.T = self.args.T  # Temperature in K

    def plot_results(self, results_array):
        """
        Records all of the parameters of interest.
        """

        results_df = pd.DataFrame(data=results_array, columns=["Interlayer mole fraction",
                                                               "Nanopore mole fraction",
                                                               "Total mole fraction",
                                                               "Chemical potential",
                                                               "Partial molar entropy",
                                                               "dq/de",
                                                               "Partial molar enthalpy"])

        uid = str(uuid.uuid1())  # Sets a unique id for the naming of files.
        cwd = os.getcwd()
        path = cwd + "/results"
        Path(path).mkdir(parents=True, exist_ok=True)
        results_df.to_csv(path + "/Na_monte_carlo_results_a_" + uid + ".csv", index=False)

        parameter_file = open(path + "/Input_arguments_a_" + uid + ".txt", "w")
        parameter_file.write(str(self.args))
        parameter_file.close()

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
            eps1 = self.args.eps1 + self.args.a * np.exp(-self.args.b*n1**self.args.c)
            return eps1 * N1 + nano_term
        else:
            u = 0
            for occ, eps in zip(self.lattice[0], self.lattice_energy[0]):
                u += occ * eps
            for occ, eps in zip(self.lattice[1], self.lattice_energy[1]):
                u += occ * eps
            return u

    def site_h(self, mu, lattice_index, random_index):
        if self.args.distribution == 4:
            if lattice_index == 0:  # Interlayers so pore term will cancel
                N1 = np.sum(self.lattice[0])  # Number of filled interlayer sites
                n1 = N1/self.lattice1_sites  # Mole fraction of interlayers
                eps1 = self.args.eps1 + self.args.a * np.exp(-self.args.b*n1**self.args.c)
                return eps1 * N1 - self.lattice[lattice_index][random_index] * mu
            else:  # Pores so interlayer term will cancel
                N2 = np.sum(self.lattice[1])  # Number of filled nanopores
                n2 = N2/self.lattice2_sites  # Mole fraction of nanopores
                M2 = self.lattice2_sites
                nano_term = (self.eps2 * n2 * M2) + (self.args.g2 * M2 * (n2 ** 2)) + (self.args.g3 * M2 * (n2 ** 3))
                return nano_term - self.lattice[lattice_index][random_index] * mu
        else:
            return self.lattice[lattice_index][random_index] * self.lattice_energy[lattice_index][random_index] - self.lattice[lattice_index][random_index] * mu

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
                # Perform a swap - Overall we didn't swap the occupation number.
                if self.lattice[lattice_index][random_index] == 0:
                    self.lattice[lattice_index][random_index] = 1
                else:
                    self.lattice[lattice_index][random_index] = 0

    def run_simulation(self):
        rows = len(self.args.mu_list) - 1  # The number of inputs - in this case the number of mu values passed through.
        row_count = 0
        results_array = np.zeros((rows + 1, 7))  # Columns for delta S, x, mu, enthalpy, etc.

        steps = self.args.sps * self.args.sites
        average_rows = steps / self.args.sample_frequency
        data_array = np.zeros((6, int(average_rows)))  # Records the parameters at the sample frequency

        for count, mu in enumerate(self.args.mu_list):
            print("Chemical potential:", mu)

            # Runs equilibration steps
            for i in range(steps):
                self.monte_carlo(mu)

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

            delta_entropy = (cov_un / variance) - mu
            results_array[row_count, 0] = mean_x1
            results_array[row_count, 1] = mean_x2
            results_array[row_count, 2] = mean_x
            results_array[row_count, 3] = mu * -1
            results_array[row_count, 4] = delta_entropy  # Partial molar entropy
            results_array[row_count, 5] = variance / (self.kb * self.T * self.args.sites)  # dq/de
            results_array[row_count, 6] = cov_un / variance  # Partial molar enthalpy
            row_count += 1

        self.plot_results(results_array)


if __name__ == '__main__':
    mc = MonteCarlo()
    mc.run_simulation()
