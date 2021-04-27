"""
Simulates the lithium intercalation in a lithium manganese oxide cathode using Monte Carlo simulation methods.
"""

import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import integrate
import time


class MonteCarlo:
    def __init__(self):
        self.number_of_mcs = 50000  # Number of monte carlo steps (MCS)
        self.start_mu = -4.3  # Start chemical potential
        self.end_mu = - 3.88  # Final chemical potential
        self.step_size_mu = 0.02  # Step size for chemical potential

        self.thermal_iterations = 20000  # Number of MCS for thermodynamic fluctuations
        self.sample_frequency = 100  # After this many MCS data is extracted

        # self.j1 = 37.5e-3  # Attraction parameter in eV of nearest
        # self.j2 = -4.0e-3  # Attraction parameter in eV of next nearest
        self.j1 = 0  # Ideal case
        self.j2 = 0  # Ideal case
        self.eps = 4.12  # In eV
        self.T = 298  # Temperature in K
        self.kb = 8.617e-5  # Boltzmann constant in eV/K

        self.grid_length = 10  # Number of unit cells containing 8 Li cells
        self.sites = (self.grid_length ** 3) * 8  # Number of sites on the cubic lattice

        self.xl = self.grid_length  # Number of atoms on the x axis
        self.yl = self.grid_length * 2  # Number of atoms on the y axis
        self.zl = self.grid_length * 4  # Number of atoms on the z axis

        self.nn_vector11 = np.array([[0, 0, -1],  # For i = 1
                                     [1, 1, -1],
                                     [1, 0, 1],
                                     [0, 1, 1]])
        self.nn_vector12 = np.array([[0, 0, 1],  # For i = -1
                                     [-1, -1, 1],
                                     [-1, 0, -1],
                                     [0, -1, -1]])
        self.nn_vector21 = np.array([[0, 0, 1],  # For i = 1
                                     [0, 0, -1],
                                     [0, 1, -1],
                                     [0, 1, 1]])
        self.nn_vector22 = np.array([[0, 0, -1],  # For i = -1
                                     [0, 0, 1],
                                     [0, -1, 1],
                                     [0, -1, -1]])

        self.lattice = np.zeros((self.xl, self.yl,
                                 self.zl))  # Initialises the lattice with all sites empty represented by a 0. A filled lattice is represented by a 1

    def get_nn_vector(self, x, y, z):
        if z % 2 == 0:
            # Sub lattice 1
            if (y + (z / 2)) % 2 == 0:
                nn_vector = self.nn_vector12
            else:
                nn_vector = self.nn_vector22
        else:
            # Sub lattice 2
            if (y + ((z + 1) / 2)) % 2 == 0:
                nn_vector = self.nn_vector11
            else:
                nn_vector = self.nn_vector21

        # nn vector contains the relative positions of all 4 nearest neighbours and depends on the sites position.
        return nn_vector

    def get_nnn_vector(self, x, y, z):
        if z % 2 == 0:
            # Sub lattice 1
            if (y + (z / 2)) % 2 == 0:
                self.i2 = -1
            else:
                self.i2 = 1
        else:
            # Sub lattice 2
            if (y + ((z + 1) / 2)) % 2 == 0:
                self.i2 = 1
            else:
                self.i2 = -1

        # This vector contains the relative positions of all 12 next nearest neighbours.
        nnn_vector = np.array([[0, - 1, 0],
                               [0, 1, 0],
                               [self.i2, 1, 0],
                               [self.i2, - 1, 0],
                               [self.i2, 0, 2],
                               [0, 0, 2],
                               [0, 1, 2],
                               [0, - 1, 2],
                               [self.i2, 0, - 2],
                               [0, 0, - 2],
                               [0, 1, - 2],
                               [0, - 1, - 2]])

        return nnn_vector

    def swap_c(self, x, y, z):
        """
        Swaps the occupation number of an atom.
        """
        if self.lattice[x, y, z] == 0:
            self.lattice[x, y, z] = 1
        else:
            self.lattice[x, y, z] = 0

    def get_nn(self, x, y, z):
        """
        Calculates the number of Li-Li interactions on a atoms nearest neighbours at position x, y, z.
        """
        sum_of_nn = 0
        nn_vector = self.get_nn_vector(x, y, z)

        # Try and except block used in case there is an index referenced out of range then the program doesnt quit.
        try:
            for vector in nn_vector:
                sum_of_nn += self.lattice[x, y, z] * self.lattice[
                    (x + vector[0]) % self.xl, (y + vector[1]) % self.yl, (z + vector[2]) % self.zl]
        except Exception as e:
            print(e)
            print("nn neighbour error: Error occurred at index: ", x, y, z)

        return sum_of_nn

    def get_nnn(self, x, y, z):
        """
        Calculates the number of Li-Li interactions on a atoms next nearest neighbours at position x, y, z.
        """
        sum_of_nnn = 0
        nnn_vector = self.get_nnn_vector(x, y, z)

        try:
            for vector in nnn_vector:
                sum_of_nnn += self.lattice[x, y, z] * self.lattice[
                    (x + vector[0]) % self.xl, (y + vector[1]) % self.yl, (z + vector[2]) % self.zl]
        except Exception as e:
            # This will occur when the index is out of range od the grid.
            print(e)
            print("Error occurred at index: ", x, y, z)

        return sum_of_nnn

    def node_hamiltonian(self, x, y, z, mu):
        """
        Returns the hamiltonian for an atom at position x, y, z and chemical potential mu.
        """

        return self.j1 * self.get_nn(x, y, z) + self.j2 * self.get_nnn(x, y, z) - self.lattice[x, y, z] * (
                self.eps + mu)  # This is the hamiltonian of 1 atom

    def node_u(self, x, y, z):
        """
        Returns the internal energy for an atom at position x, y, z.
        """

        return self.j1 * self.get_nn(x, y, z) + self.j2 * self.get_nnn(x, y, z) - self.lattice[x, y, z] * (
            self.eps)  # This is the internal energy of 1 atom

    def monte_carlo(self, mu):
        """
        Selects a random position in the lattice and swaps the occupation number to see if it lowers the energy.
        Accepts if it lowers the energy or under the boltzmann distribution.
        """

        random_position = [random.randint(0, self.xl - 1), random.randint(0, self.yl - 1),
                           random.randint(0, self.zl - 1)]
        current_h = self.node_hamiltonian(random_position[0], random_position[1], random_position[2], mu)

        self.swap_c(random_position[0], random_position[1], random_position[2])

        trial_h = self.node_hamiltonian(random_position[0], random_position[1], random_position[2], mu)
        delta_h = current_h - trial_h

        if delta_h < 0:
            # Only keep the swap if a random float between 0 and 1 is <= P
            rand_p = random.random()  # Generates a random number between 0 and 1
            p = math.exp(-abs(delta_h) / (self.kb * self.T))
            if rand_p > p:
                # Don't accept change and reverse swap. So overall no swap is made
                self.swap_c(random_position[0], random_position[1], random_position[2])

    def lattice_h(self, mu):
        """
        :return the hamiltonian for the entire lattice.
        """

        h = 0
        for z in range(self.zl):
            for y in range(self.yl):
                for x in range(self.xl):
                    h += self.node_hamiltonian(x, y, z, mu)

        return h

    def lattice_u(self):
        """
        :return: the internal energy for the entire lattice
        """

        u = 0
        for z in range(self.zl):
            for y in range(self.yl):
                for x in range(self.xl):
                    u += self.node_u(x, y, z)

        return u

    def lattice_n(self):
        """
        :return: the total number of Li atoms in the lattice
        """

        return np.sum(self.lattice)

    def mole_fraction(self):
        """
        :return the lattice mole fraction and sublattice mole fractions and prints to the console.
        """

        sl1_n = 0
        sl2_n = 0

        for z in range(self.zl):
            for y in range(self.yl):
                for x in range(self.xl):
                    if z % 2 == 0:
                        sl1_n += self.lattice[x, y, z]
                    else:
                        sl2_n += self.lattice[x, y, z]

        lattice_n = sl1_n + sl2_n
        lattice_x = lattice_n / self.sites
        print("Li mole fraction in sub lattice 1:", sl1_n / (self.sites / 2),
              " Li mole fraction in sub lattice 2: ",
              sl2_n / (self.sites / 2))
        print("Total Li atoms:", lattice_n, " Total sites:", self.sites, " Total mole fraction:",
              lattice_x)

        return lattice_x, sl1_n / (self.sites / 2), sl2_n / (self.sites / 2)

    def thermal_averaging(self, mu):
        """
        :return: The variance and covariance for a set of samples
        """

        total_rows = self.thermal_iterations / self.sample_frequency
        data_array = np.zeros((int(total_rows), 4))  # Columns for U, N, UN and N^2.
        row_count = 0

        for i in range(self.thermal_iterations):
            self.monte_carlo(mu)
            if i % self.sample_frequency == 0:
                u = self.lattice_u()
                n = self.lattice_n()
                u_n = u * n
                n_n = n * n

                data_array[row_count, 0] = u
                data_array[row_count, 1] = n
                data_array[row_count, 2] = u_n
                data_array[row_count, 3] = n_n
                row_count += 1

        df = pd.DataFrame(data=data_array, columns=["U", "N", "UN", "NN"])
        mean_u = df["U"].mean()
        mean_n = df["N"].mean()
        mean_un = df["UN"].mean()
        mean_nn = df["NN"].mean()
        variance = mean_nn - mean_n ** 2
        cov_un = mean_un - (mean_u * mean_n)

        return variance, cov_un

    def run_simulation(self):
        """
        Runs the main loop through the range of chemical potential values and saves results to numpy array.
        """

        rows = math.ceil((self.end_mu - self.start_mu) / self.step_size_mu)
        row_count = 0
        results_array = np.zeros(
            (rows + 1, 6))  # Columns for delta S, x, mu, sl1 mole fraction, sl2 mole fraction and entropy

        for mu in np.arange(self.start_mu, self.end_mu + self.step_size_mu, self.step_size_mu):
            startMC_time = time.time()
            print("Chemical potential: ", mu)
            for i in range(self.number_of_mcs):
                self.monte_carlo(mu)
            finishMC_time = time.time()
            print("Time to equilibrium: ", finishMC_time - startMC_time)
            total_mole_fraction, sl1_mole_fraction, sl2_mole_fraction = self.mole_fraction()  # Prints the final mole fraction

            var, cov = self.thermal_averaging(mu)
            finishThermo_time = time.time()
            print("Time to thermo average: ", finishMC_time - finishThermo_time)

            delta_entropy = (1 / self.T) * (cov / var - mu)
            results_array[row_count, 0] = delta_entropy
            results_array[row_count, 1] = total_mole_fraction
            results_array[row_count, 2] = -1 * mu
            results_array[row_count, 3] = sl1_mole_fraction
            results_array[row_count, 4] = sl2_mole_fraction
            results_array[row_count, 5] = var / (self.kb * self.T * self.sites)
            row_count += 1
            finishData_time = time.time()
            print("Time to add data: ", finishData_time - finishThermo_time)
            print("----------")

        self.run_results(results_array)

    def run_results(self, results_array):
        """
        Presents the results from the simulation.
        """

        results_dataframe = pd.DataFrame(data=results_array, columns=["Delta Entropy", "Mole Fraction of Li",
                                                                      "Chemical potential",
                                                                      "Mole fraction sub lattice 1",
                                                                      "Mole fraction sub lattice 2", "dq/dv"])

        entropy_list = integrate.cumtrapz(results_array[:, 0], results_array[:, 1],
                                          initial=0)  # Contains the entropy values
        results_dataframe['Entropy'] = entropy_list

        cwd = os.getcwd()
        path = cwd + "/monte_carlo_results.csv"
        results_dataframe.to_csv(path, index=False)

        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 2

        fig.suptitle('Number of MCS: %i' % self.number_of_mcs)

        ax1 = results_dataframe.plot(kind='scatter', ax=axes[0, 0], x='Mole Fraction of Li', y='Chemical potential',
                                     color='black', s=4)
        ax2 = results_dataframe.plot(kind='scatter', ax=axes[0, 1], x='Mole Fraction of Li', y='Delta Entropy',
                                     color='black', s=4)
        ax3 = results_dataframe.plot(kind='scatter', ax=axes[1, 0], x='Mole Fraction of Li',
                                     y='Mole fraction sub lattice 1', color='red', s=4)
        results_dataframe.plot(kind='scatter', ax=ax3, x='Mole Fraction of Li', y='Mole fraction sub lattice 2',
                               color='blue', s=4)
        ax4 = results_dataframe.plot(kind='scatter', ax=axes[1, 1], x='Mole Fraction of Li', y='Entropy', color='black',
                                     s=4)

        ax3.legend(['Sublattice 1', 'Sublattice 2'])

        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
        ax3.set_xlim([0, 1])
        ax4.set_xlim([0, 1])

        ax3.set_xlabel('x')
        ax4.set_xlabel('x')

        ax1.set_ylabel('E / V vd. Li/Li+')
        ax2.set_ylabel('dS/dx [kJ/mol K]')
        ax3.set_ylabel('Sublattice occupancy')
        ax4.set_ylabel('S [kJ/mol K]')

        fig.savefig(cwd + "/results_plot.png")
        plt.show()


if __name__ == '__main__':
    start_time = time.time()

    mc = MonteCarlo()
    mc.run_simulation()

    end_time = time.time()
    print("Execution time: ", end_time - start_time)
