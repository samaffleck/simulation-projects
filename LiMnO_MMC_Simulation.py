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
import argparse


class Atom:
    """
    Atom class contains the object of a single Li atom
    """

    def __init__(self, x_index, y_index, z_index, grid_length, j1, j2, eps):

        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index

        self.j1 = j1  # Attraction parameter in eV of nearest
        self.j2 = j2  # Attraction parameter in eV of next nearest
        self.eps = eps  # In eV

        self.xl = grid_length  # Number of atoms in the x axis
        self.yl = grid_length * 2  # Number of atoms in the y axis
        self.zl = grid_length * 4  # Number of atoms in the z direction

        if self.z_index % 2 == 0:
            # Sub lattice 1
            self.i = -1
        else:
            # Sub lattice 2
            self.i = 1

        # Vectors 1 and 2 index the position in the lattice of any nodes nearest neighbour.
        # They depend on the sub lattice the atom is in and also their z and y index.
        # Each sub lattice has 2 potential vectors that describe the index of their nearest neighbours.
        # The are similar vectors and the value self.i accounts for this difference between the vectors.
        self.nn_vector1 = np.array([[self.x_index, self.y_index, (self.z_index - self.i) % self.zl],
                                    [(self.x_index + self.i) % self.xl, (self.y_index + self.i) % self.yl, (self.z_index - self.i) % self.zl],
                                    [(self.x_index + self.i) % self.xl, self.y_index, (self.z_index + self.i) % self.zl],
                                    [self.x_index, (self.y_index + self.i) % self.yl, (self.z_index + self.i) % self.zl]])

        self.nn_vector2 = np.array([[self.x_index, self.y_index, (self.z_index + self.i) % self.zl],
                                    [self.x_index, self.y_index, (self.z_index - self.i) % self.zl],
                                    [self.x_index, (self.y_index + self.i) % self.yl, (self.z_index - self.i) % self.zl],
                                    [self.x_index, (self.y_index + self.i) % self.yl, (self.z_index + self.i) % self.zl]])

        if self.z_index % 2 == 0:
            # Sub lattice 1
            if (self.y_index + (self.z_index / 2)) % 2 == 0:
                self.nn_vector = self.nn_vector1
                self.x_offset = 0
                self.i2 = -1
            else:
                self.nn_vector = self.nn_vector2
                self.x_offset = 0.5
                self.i2 = 1
        else:
            # Sub lattice 2
            if (self.y_index + ((self.z_index + 1) / 2)) % 2 == 0:
                self.nn_vector = self.nn_vector1
                self.x_offset = 0.5
                self.i2 = 1
            else:
                self.nn_vector = self.nn_vector2
                self.x_offset = 0
                self.i2 = -1

        # This vector contains the relative positions of all 12 next nearest neighbours.
        self.nnn_vector = np.array([[self.x_index, self.y_index - 1, self.z_index],
                                    [self.x_index, (self.y_index + 1) % self.yl, self.z_index],
                                    [(self.x_index + self.i2) % self.xl, (self.y_index + 1) % self.yl, self.z_index],
                                    [(self.x_index + self.i2) % self.xl, self.y_index - 1, self.z_index],
                                    [(self.x_index + self.i2) % self.xl, self.y_index, (self.z_index + 2) % self.zl],
                                    [self.x_index, self.y_index, (self.z_index + 2) % self.zl],
                                    [self.x_index, (self.y_index + 1) % self.yl, (self.z_index + 2) % self.zl],
                                    [self.x_index, self.y_index - 1, (self.z_index + 2) % self.zl],
                                    [(self.x_index + self.i2) % self.xl, self.y_index, self.z_index - 2],
                                    [self.x_index, self.y_index, self.z_index - 2],
                                    [self.x_index, (self.y_index + 1) % self.yl, self.z_index - 2],
                                    [self.x_index, self.y_index - 1, self.z_index - 2]])

        self.c = 0  # The occupation number, c = 1 is a Lithium atom c = 0 is a vacant site.
        self.temp_neighbours = []  # Temporarily stores the neighbours before converting to a numpy array.

    def __add__(self, other):
        if isinstance(other, Atom):
            return self.c + other.c
        else:
            return self.c + other

    def __radd__(self, other):
        return self + other

    def swap_c(self):
        """
        Swaps the occupation number of an atom.
        """
        if self.c == 0:
            self.c = 1
        else:
            self.c = 0

    def get_position(self):
        """
        Gets the cartesian coordinates for the atom in the lattice
        """

        if self.z_index % 2 != 0:
            self.offset_vector = (0.25, 0.25, 0)
        else:
            self.offset_vector = (0, 0, 0)

        self.position = (self.x_index + self.x_offset + self.offset_vector[0], self.y_index * 0.5 +
                         self.offset_vector[1], self.z_index * 0.25 + self.offset_vector[2])

    def get_nn(self, grid):
        self.temp_neighbours = []

        for vector in self.nn_vector:
            self.temp_neighbours.append(grid[vector[0]][vector[1]][vector[2]])

        self.neighbours = np.array(self.temp_neighbours)

    def get_nnn(self, grid):
        self.temp_neighbours = []

        for vector in self.nnn_vector:
            self.temp_neighbours.append(grid[vector[0]][vector[1]][vector[2]])

        self.nnn = np.array(self.temp_neighbours)

    def node_hamiltonian(self, mu):
        sum_of_nn = 0
        sum_of_nnn = 0

        if self.c == 1:

            # sum_of_nn = np.sum(self.neighbours)
            for nn in self.neighbours:
                sum_of_nn += (self.c * nn.c)
            nn_term = sum_of_nn * self.j1

            # sum_of_nnn = np.sum(self.nnn)
            for nnn in self.nnn:
                sum_of_nnn += (self.c * nnn.c)
            nnn_term = sum_of_nnn * self.j2
        else:
            return 0, 0

        chemical_potential_term = - self.c * (self.eps + mu)
        potential_term = - self.c * self.eps

        return nn_term + nnn_term + chemical_potential_term, nn_term + nnn_term + potential_term

    def get_node_internal_energy(self):
        sum_of_nn = 0
        sum_of_nnn = 0

        if self.c == 1:
            for nn in self.neighbours:
                sum_of_nn += (self.c * nn.c)
            nn_term = sum_of_nn * self.j1

            for nnn in self.nnn:
                sum_of_nnn += (self.c * nnn.c)
            nnn_term = sum_of_nnn * self.j2
        else:
            nn_term = 0
            nnn_term = 0

        potential_term = - self.c * self.eps

        return nn_term + nnn_term + potential_term


def get_grid(grid_length, j1, j2, eps):
    """
    Get_grid() returns a 3 dimensional array of 'Atom' objects and is representative of the lattice structure.
    """

    # The 3 dimensional array that will store each Li atom with shape (1, 2, 4)
    grid = np.zeros((grid_length, grid_length * 2, grid_length * 4), dtype=Atom)

    # Initialise the grid.
    for z in range(
            grid_length * 4):  # Even z layers/indexes are in sub lattice 1 and odd z layers are in sub lattice 2.
        for y in range(grid_length * 2):
            for x in range(grid_length):
                grid[x][y][z] = Atom(x, y, z, grid_length, j1, j2, eps)

    # Initialise the neighbours for all atoms
    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                grid[x][y][z].get_nn(grid)
                grid[x][y][z].get_nnn(grid)

    return grid


def get_internal_energy(grid, grid_length):
    """
    Calculates the total internal energy
    """

    internal_energy = 0
    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                internal_energy += grid[x][y][z].get_node_internal_energy()

    return internal_energy


def get_occupancy(grid, gird_length):
    occupancy = 0
    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                occupancy += grid[x][y][z].c

    return occupancy


def get_mole_fraction(grid, grid_length):
    """
    Returns the mole fraction of lithium atoms in each sub lattice.
    """
    li_count_sl1 = 0
    li_count_sl2 = 0

    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                if z % 2 == 0:
                    li_count_sl1 += grid[x][y][z].c
                else:
                    li_count_sl2 += grid[x][y][z].c

    total_num_of_li = li_count_sl1 + li_count_sl2
    total_num_of_sites = (grid_length ** 3) * 8
    total_mole_fraction = total_num_of_li / total_num_of_sites
    print("Li mole fraction in sub lattice 1:", li_count_sl1/(total_num_of_sites/2), " Li mole fraction in sub lattice 2: ",
          li_count_sl2/(total_num_of_sites/2))
    print("Total Li atoms:", total_num_of_li, " Total sites:", total_num_of_sites, " Total mole fraction:",
          total_mole_fraction)

    return total_mole_fraction, li_count_sl1/(total_num_of_sites/2), li_count_sl2/(total_num_of_sites/2)


def plot_results(results_array, number_of_mcs):

    results_dataframe = pd.DataFrame(data=results_array, columns=["Delta Entropy", "Mole Fraction of Li",
                                                                  "Chemical potential", "Mole fraction sub lattice 1",
                                                                  "Mole fraction sub lattice 2", "dq/de", "Partial molar enthalpy"])

    entropy_list = integrate.cumtrapz(results_array[:, 0], results_array[:, 1], initial=0)  # Contains the entropy values
    results_dataframe['Entropy'] = entropy_list
    results_dataframe['Order parameter'] = abs(results_dataframe['Mole fraction sub lattice 1'] - results_dataframe['Mole fraction sub lattice 2'])

    cwd = os.getcwd()
    path = cwd + "/monte_carlo_results.csv"
    results_dataframe.to_csv(path, index=False)

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2

    fig.suptitle('Number of MCS: %i' % number_of_mcs)

    ax1 = results_dataframe.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes[0, 0], x='Mole Fraction of Li', y='Chemical potential')
    ax2 = results_dataframe.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes[0, 1], x='Mole Fraction of Li', y='Delta Entropy')
    ax3 = results_dataframe.plot(linestyle='-', color='blue', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes[1, 0], x='Mole Fraction of Li',
                                 y='Mole fraction sub lattice 1')
    results_dataframe.plot(linestyle='-', color='green', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=ax3, x='Mole Fraction of Li', y='Mole fraction sub lattice 2')
    ax4 = results_dataframe.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes[1, 1], x='Mole Fraction of Li', y='Entropy')

    #ax3.legend(['Sub lattice 1', 'Sub lattice 2'])

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

    fig2, axes2 = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig2.suptitle('Number of MCS: %i' % number_of_mcs)

    ax5 = results_dataframe.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes2[0, 0], x='Chemical potential', y='dq/de')
    ax6 = results_dataframe.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes2[0, 1], x='Mole Fraction of Li', y='Order parameter')
    ax7 = results_dataframe.plot(linestyle='-', color='black', lw=0.5, marker='o', markeredgecolor='black', markersize=4, ax=axes2[1, 0], x='Mole Fraction of Li', y='Partial molar enthalpy')

    ax5.set_xlabel('E ')
    ax6.set_xlim([0, 1])
    ax6.set_xlabel('x')
    ax7.set_xlim([0, 1])
    ax7.set_xlabel('x')

    ax5.set_ylabel('- dx/dV')
    ax6.set_ylabel('|n1-n2|')
    ax6.set_ylabel('Partial molar enthalpy')

    plt.show()


def thermal_fluctuations(grid, grid_length, kb, T, mu, mcs, sample_frequency):
    total_iterations = mcs
    total_rows = total_iterations / sample_frequency
    data_array = np.zeros((int(total_rows), 4))  # Columns for U, N, UN and N^2.
    row_count = 0
    occupancy = get_occupancy(grid, grid_length)  # Get the initial occupancy by looping through the whole grid.
    internal_energy = get_internal_energy(grid, grid_length)

    for i in range(total_iterations):
        grid, atom_occ, swapped, new_u, random_atom_u = monte_carlo(grid, grid_length, kb, T, mu, True)
        if swapped:
            if atom_occ == 0:
                occupancy += 1
            else:
                occupancy -= 1
            internal_energy += new_u - random_atom_u

        if i % sample_frequency == 0:
            internal_energy_times_occupancy = internal_energy * occupancy
            occupancy_squared = occupancy * occupancy
            data_array[row_count, 0] = internal_energy
            data_array[row_count, 1] = occupancy
            data_array[row_count, 2] = internal_energy_times_occupancy
            data_array[row_count, 3] = occupancy_squared
            row_count += 1

    df = pd.DataFrame(data=data_array, columns=["U", "N", "UN", "NN"])
    mean_u = df["U"].mean()
    mean_n = df["N"].mean()
    mean_un = df["UN"].mean()
    mean_nn = df["NN"].mean()
    variance = mean_nn - mean_n ** 2
    cov_un = mean_un - (mean_u * mean_n)

    return variance, cov_un


def monte_carlo(grid, grid_length, kb, T, mu, update_occ=False):
    """
    This method contains the metropolis monte carlo algorithm and is classed as 1 monte carlo step.
    """
    rand_atom = grid[random.randint(0, grid_length - 1)][random.randint(0, (grid_length * 2) - 1)][
        random.randint(0, (grid_length * 4) - 1)]
    rand_atom_occ = rand_atom.c

    current_h, random_atom_u = rand_atom.node_hamiltonian(mu)
    rand_atom.swap_c()
    swapped = True
    new_h, new_u = rand_atom.node_hamiltonian(mu)
    delta_h = current_h - new_h

    if delta_h < 0:
        # Only keep the swap if a random float between 0 and 1 is <= P
        rand_p = random.random()  # Generates a random number between 0 and 1
        p = math.exp(-abs(delta_h) / (kb * T))
        if rand_p > p:
            # Don't accept change and reverse swap. So overall no swap is made
            rand_atom.swap_c()
            swapped = False

    if update_occ:
        return grid, rand_atom_occ, swapped, new_u, random_atom_u
    else:
        return grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Monte carlo simulation')
    parser.add_argument('--grid_length', type=int, metavar='', help='Number of unit cells', default=10)
    parser.add_argument('--temperature', type=int, metavar='', help='Temperature', default=298)
    parser.add_argument('--eps', type=float, metavar='', help='Epsilon value ', default=4.12)
    parser.add_argument('--j1', type=float, metavar='', help='J1 interaction parameter for nearest neighbours', default=37.5e-3)
    parser.add_argument('--j2', type=float, metavar='', help='J2 interaction parameter for next nearest parameters', default=-4.0e-3)
    parser.add_argument('--mcs', type=int, metavar='', help='Number of monte carlo steps', default=500000)
    parser.add_argument('--step', type=float, metavar='', help='Chemical potential step size', default=0.01)
    parser.add_argument('--frequency', type=int, metavar='', help='Number of mcs before a sample is taken', default=200)
    args = parser.parse_args()

    kb = 8.617e-5  # Boltzmann constant in eV/K
    grid_length = args.grid_length  # This is the number of unit cells - contains 8 lithium atoms.
    lattice_points = (grid_length ** 3) * 8
    T = args.temperature  # Temperature in K
    eps = args.eps  # in eV
    j1 = args.j1  # Attraction parameter in eV of nearest
    j2 = args.j2  # Attraction parameter in eV of next nearest

    grid = get_grid(grid_length, j1, j2, eps)  # Returns a numpy 3d array [x][y][z] of each primitive cell.

    number_of_mcs = args.mcs
    sample_frequency = args.frequency
    start_mu = -4.3  # Start chemical potential
    end_mu = - 3.88
    step_size_mu = args.step  # Step size for chemical potential
    rows = math.ceil((end_mu - start_mu) / step_size_mu)
    row_count = 0
    results_array = np.zeros((rows + 1, 7))  # Columns for delta S, x, mu, sl1 mole fraction, sl2 mole fraction

    for chem in np.arange(start_mu, end_mu + step_size_mu, step_size_mu):
        print("Chemical potential: ", chem)
        startMC_time = time.time()
        for i in range(number_of_mcs):
            grid = monte_carlo(grid, grid_length, kb, T, chem)

        finishMC_time = time.time()
        print("Time to finish MC steps: ", finishMC_time - startMC_time)
        total_mole_fraction, sl1_mole_fraction, sl2_mole_fraction = get_mole_fraction(grid, grid_length)  # Prints the final mole fraction

        var, cov = thermal_fluctuations(grid, grid_length, kb, T, chem, number_of_mcs, sample_frequency)

        finishThermo_time = time.time()
        print("Time to finish thermo averaging:", finishThermo_time - finishMC_time)

        delta_entropy = (1 / T) * ((cov / var) - chem)
        results_array[row_count, 0] = delta_entropy
        results_array[row_count, 1] = total_mole_fraction
        results_array[row_count, 2] = -1 * chem
        results_array[row_count, 3] = sl1_mole_fraction
        results_array[row_count, 4] = sl2_mole_fraction
        results_array[row_count, 5] = var/(kb * T * lattice_points)
        results_array[row_count, 6] = cov / var  # Partial molar enthalpy
        row_count += 1
        print("----------")

    plot_results(results_array, number_of_mcs)
