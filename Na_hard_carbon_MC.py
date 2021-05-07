"""
Monte carlo simulation for the insertion of sodium into hard carbon.
"""

import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


class MonteCarlo:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monte carlo simulation for sodation into hard carbon")
        self.parser.add_argument('--sites', type=int, metavar='', help='Total number of sites including nanopores and interlayers', default=500)
        self.parser.add_argument('--l', type=float, metavar='', help='Fraction of total sites in the interlayers', default=0.3)
        self.parser.add_argument('--mcs', type=int, metavar='', help='Number of monte carlo steps', default=20000)
        self.parser.add_argument('--eps1', type=float, metavar='', help='Point interaction term for interlayers in eV', default=-0.377)
        self.parser.add_argument('--delE', type=float, metavar='', help='Point term for nanopores - priori heterogeneity in eV', default=-0.25)
        self.parser.add_argument('--g2', type=float, metavar='', help='g2 term', default=0)
        self.parser.add_argument('--g3', type=float, metavar='', help='g3 term', default=0)
        self.parser.add_argument('--a', type=float, metavar='', help='a term', default=0.0)
        self.parser.add_argument('--b', type=float, metavar='', help='b term', default=1.5)
        self.parser.add_argument('--c', type=float, metavar='', help='c term', default=1.0)
        self.parser.add_argument('--sample_frequency', type=int, metavar='', help='Number of mcs before a sample is taken', default=200)
        self.parser.add_argument('--start_mu', type=float, metavar='', help='starting chemical potential', default=-0.45)
        self.parser.add_argument('--finish_mu', type=float, metavar='', help='starting chemical potential', default=0.0)
        self.parser.add_argument('--step_chem', type=float, metavar='', help='step size of chemical potential', default=0.01)

        self.args = self.parser.parse_args()

        self.eps1 = self.args.eps1
        self.eps2 = self.args.eps1 - self.args.delE

        self.lattice1_sites = int(self.args.sites * self.args.l)  # Number of sites in the interlayers.
        self.lattice2_sites = self.args.sites - self.lattice1_sites  # Number of sites in the nanopores.
        # self.lattice1 = np.zeros(self.lattice1_sites)  # Interlayers where 0 is an empty site and 1 is a filled Na site.
        # self.lattice2 = np.zeros(self.lattice2_sites)  # Nanopores.
        self.lattice = [np.zeros(self.lattice1_sites), np.zeros(self.lattice2_sites)]

        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.T = 288  # Temperature in K

        self.start_mu = self.args.start_mu
        self.end_mu = self.args.finish_mu
        self.step_size_mu = self.args.step_chem

    def calculate_h(self, mu):
        """
        Calculates the hamiltonian.
        """
        N1 = np.sum(self.lattice[0])
        N2 = np.sum(self.lattice[1])
        # n1 = N1/self.lattice1_sites
        # n2 = N2/self.lattice2_sites
        # m1 = self.lattice1_sites
        # m2 = self.lattice2_sites
        # eps1_prime = self.eps1 + self.args.a * math.exp(-self.args.b * n1 ** self.args.c)

        h = self.eps1 * N1 + self.eps2 * N2 - (N1 + N2) * mu

        return h

    def calculate_u(self):
        N1 = np.sum(self.lattice[0])
        N2 = np.sum(self.lattice[1])
        u = self.eps1 * N1 + self.eps2 * N2

        return u

    def plot_results(self, results_array):
        results_df = pd.DataFrame(data=results_array, columns=["Interlayer mole fraction",
                                                               "Nanopore mole fraction",
                                                               "Total mole fraction",
                                                               "Chemical potential",
                                                               "Partial molar entropy",
                                                               "dq/de",
                                                               "Partial molar enthalpy"])
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

        ax1 = results_df.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        ax2 = results_df.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        ax3 = results_df.plot(linestyle='-', color='blue', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[1, 0], x='Total mole fraction', y='Interlayer mole fraction')
        results_df.plot(linestyle='-', color='green', lw=0.5, marker='o', markeredgecolor='black', markersize=4,
                        ax=ax3, x='Total mole fraction', y='Nanopore mole fraction')
        ax4 = results_df.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')

        plt.show()

    def monte_carlo(self, mu):
        rand_l = random.random()
        if rand_l < self.args.l:
            lattice_index = 0
        else:
            lattice_index = 1

        random_index = random.randint(0, np.size(self.lattice[lattice_index]) - 1)

        current_h = self.calculate_h(mu)
        # Perform a swap
        if self.lattice[lattice_index][random_index] == 0:
            self.lattice[lattice_index][random_index] = 1
        else:
            self.lattice[lattice_index][random_index] = 0

        new_h = self.calculate_h(mu)
        delta_h = new_h - current_h

        if delta_h > 0:
            rand_p = random.random()
            p = math.exp(-delta_h / (self.kb * self.T))
            if rand_p > p:
                # Perform a swap
                if self.lattice[lattice_index][random_index] == 0:
                    self.lattice[lattice_index][random_index] = 1
                else:
                    self.lattice[lattice_index][random_index] = 0

    def run_simulation(self):
        rows = math.ceil((self.end_mu - self.start_mu) / self.step_size_mu)
        row_count = 0
        results_array = np.zeros((rows + 1, 7))  # Columns for delta S, x, mu, sl1 mole fraction, sl2 mole fraction

        average_rows = self.args.mcs / self.args.sample_frequency
        data_array = np.zeros((6, int(average_rows)))

        for mu in np.arange(self.start_mu, self.end_mu + self.step_size_mu, self.step_size_mu):
            mu = round(mu, 3)
            print("Chemical potential:", mu)

            # Runs equilibration steps
            for i in range(self.args.mcs):
                self.monte_carlo(mu)

            data_count = 0
            for i in range(self.args.mcs):
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
            mean_x1 = np.average(data_array[4])/self.lattice1_sites
            mean_x2 = np.average(data_array[5])/self.lattice2_sites
            mean_x = mean_n/self.args.sites
            variance = mean_nn - mean_n ** 2
            cov_un = mean_un - (mean_u * mean_n)

            print('Interlayer mole fraction:', mean_x1, 'Nanopore mole fraction:', mean_x2, 'Total mole fraction:', mean_x)
            print("--------")

            delta_entropy = (1 / self.T) * ((cov_un / variance) - mu)
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
