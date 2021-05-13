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
        self.parser.add_argument('--mcs', type=int, metavar='', help='Number of monte carlo steps', default=1000000)
        self.parser.add_argument('--eps1', type=float, metavar='', help='Point interaction term for interlayers in eV', default=-0.377)
        self.parser.add_argument('--delE', type=float, metavar='', help='Point term for nanopores - priori heterogeneity in eV', default=-0.25)
        self.parser.add_argument('--g2', type=float, metavar='', help='g2 term', default=0)
        self.parser.add_argument('--g3', type=float, metavar='', help='g3 term', default=0)
        self.parser.add_argument('--a', type=float, metavar='', help='a term', default=0.0)
        self.parser.add_argument('--b', type=float, metavar='', help='b term', default=1.5)
        self.parser.add_argument('--c', type=float, metavar='', help='c term', default=1.0)
        self.parser.add_argument('--sample_frequency', type=int, metavar='', help='Number of mcs before a sample is taken', default=200)
        self.parser.add_argument('--start_mu', type=float, metavar='', help='starting chemical potential', default=-1.0)
        self.parser.add_argument('--finish_mu', type=float, metavar='', help='starting chemical potential', default=0.0)
        self.parser.add_argument('--step_chem', type=float, metavar='', help='step size of chemical potential', default=0.04)
        self.parser.add_argument('--T', type=float, metavar='', help='Temperature', default=288)
        self.parser.add_argument('--eps1_max', type=float, metavar='', help='Maximum point value for interlayers', default=-0.077)
        self.parser.add_argument('--eps1_min', type=float, metavar='', help='Minimum point value for interlayers', default=-0.677)
        self.parser.add_argument('--eps1_mean', type=float, metavar='', help='Mean value for interlayers', default=-0.377)
        self.parser.add_argument('--eps1_sig', type=float, metavar='', help='Standard deviation for the point values for interlayers', default=0.1)

        self.args = self.parser.parse_args()

        self.eps1 = self.args.eps1
        self.eps2 = self.args.eps1 - self.args.delE

        self.lattice1_sites = int(self.args.sites * self.args.l)  # Number of sites in the interlayers.
        self.lattice2_sites = self.args.sites - self.lattice1_sites  # Number of sites in the nanopores.
        self.lattice = [np.zeros(self.lattice1_sites), np.zeros(self.lattice2_sites)] # Stores the occupation of all the sites in the lattice. Lattice[0] are the interlayers and lattice[1] are the nanopores.
        # self.lattice_energy = [np.random.uniform(self.args.eps1_min, self.args.eps1_max, self.lattice1_sites), np.zeros(self.lattice2_sites)]  # Stores the point energy for every site in the lattice.
        self.lattice_energy = [np.random.normal(self.args.eps1_mean, self.args.eps1_sig, self.lattice1_sites), np.zeros(self.lattice2_sites)]  # Stores the point energy for every site in the lattice.

        for i, site in enumerate(self.lattice_energy[1]):
            self.lattice_energy[1][i] = self.eps2

        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.T = self.args.T  # Temperature in K

        self.start_mu = self.args.start_mu
        self.end_mu = self.args.finish_mu
        self.step_size_mu = self.args.step_chem

    def calculate_h(self, mu):
        """
        Calculates the hamiltonian.
        """
        N1 = np.sum(self.lattice[0])
        N2 = np.sum(self.lattice[1])

        l1_h = np.sum(np.multiply(self.lattice[0], self.lattice_energy[0]))
        l2_h = np.sum(np.multiply(self.lattice[1], self.lattice_energy[1]))

        h = l1_h + l2_h - (N1 + N2) * mu  # For a random point distribution
        h2 = self.eps1 * N1 + self.eps2 * N2 - (N1 + N2) * mu  # For a constant point energy

        #print('h:', h, ' h2:', h2)
        return h

    def calculate_u(self):
        l1_u = np.sum(np.multiply(self.lattice[0], self.lattice_energy[0]))
        l2_u = np.sum(np.multiply(self.lattice[1], self.lattice_energy[1]))
        u = l1_u + l2_u

        return u

    def plot_results(self, results_array):
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 2

        results_df = pd.DataFrame(data=results_array, columns=["Interlayer mole fraction",
                                                               "Nanopore mole fraction",
                                                               "Total mole fraction",
                                                               "Chemical potential",
                                                               "Partial molar entropy",
                                                               "dq/de",
                                                               "Partial molar enthalpy"])

        cwd = os.getcwd()
        path = cwd + "/Na_monte_carlo_results.csv"
        results_df.to_csv(path, index=False)

        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

        ax1 = results_df.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        ax2 = results_df.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        #ax3 = results_df.plot(linestyle='-', color='blue', lw=0.5, marker='o', markeredgecolor='black',
        #                      markersize=4, ax=axes[1, 0], x='Total mole fraction', y='Interlayer mole fraction')
        #results_df.plot(linestyle='-', color='green', lw=0.5, marker='o', markeredgecolor='black', markersize=4,
        #                ax=ax3, x='Total mole fraction', y='Nanopore mole fraction')
        ax3 = results_df.plot(linestyle='-', color='blue', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[1, 0], x='Chemical potential', y='dq/de')
        ax4 = results_df.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black',
                              markersize=4, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')

        fig.suptitle('Number of MCS: %i \n Number of lattice points: %i' % (self.args.mcs, self.args.sites))
        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
        ax4.set_xlim([0, 1])

        ax1.set_xlabel('Na content, x')
        ax2.set_xlabel('Na content, x')
        ax3.set_xlabel('Voltage V')
        ax4.set_xlabel('Na content, x')

        ax1.set_ylabel('Voltage [V]')
        ax2.set_ylabel('dS/dx []')
        ax3.set_ylabel('dq/de')
        ax4.set_ylabel('Partial molar enthalpy []')

        parameter_file = open("Input_arguments.txt", "w")
        parameter_file.write(str(self.args))
        parameter_file.close()

        fig_path = cwd + "/Na_plot_results.png"
        plt.savefig(fig_path)
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
