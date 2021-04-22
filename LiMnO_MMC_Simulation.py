"""
Simulates the lithium intercalation in a lithium manganese oxide cathode using Monte Carlo simulation methods.
"""

import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import os


class Atom:
    """
    Atom class contains the object of a single Li atom

    ...

    Methods
    ----------
    swap_c(self)
        Swaps the occupation number for a atom where 1 represents a Li atom and 0 represents a vecent site.

    get_position(self)
        Prints the cartesian coordinates for the atom based on its index in the lattice.

    get_nn(self, grid)
        Assigns the atoms 4 nearest neighbours in the lattice.

    get_nnn(self, grid)
        Assigns the atoms 12 next nearest neighbours in the lattice.

    node_hamiltonian(self)
        Loops through the nearest and next nearest neighbours to calculate the hamiltonian for 1 atom,
        this is used to calculate the relative change in H for a given configuration change.
    """

    def __init__(self, x_index, y_index, z_index, grid_length):
        """
        Parameters
        ----------
        :param x_index:
            This is the x index in the grid/lattice
        :param y_index:
            This is the y index in the grid/lattice
        :param z_index:
            This is the z index in the grid/lattice
        :param grid_length:
            This is the number of unit cells per side of a cube that each contains 8 Li atoms.
        """

        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index

        self.j1 = 37.5e-3  # Attraction parameter in eV of nearest
        self.j2 = -4.0e-3  # Attraction parameter in eV of next nearest
        self.eps = 4.12  # In eV

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
        self.nn_vector1 = np.array([[self.x_index, self.y_index, oor_index(self.z_index - self.i, self.zl)],
                                    [oor_index(self.x_index + self.i, self.xl),
                                     oor_index(self.y_index + self.i, self.yl),
                                     oor_index(self.z_index - self.i, self.zl)],
                                    [oor_index(self.x_index + self.i, self.xl), self.y_index,
                                     oor_index(self.z_index + self.i, self.zl)],
                                    [self.x_index, oor_index(self.y_index + self.i, self.yl),
                                     oor_index(self.z_index + self.i, self.zl)]])

        self.nn_vector2 = np.array([[self.x_index, self.y_index, oor_index(self.z_index + self.i, self.zl)],
                                    [self.x_index, self.y_index, oor_index(self.z_index - self.i, self.zl)],
                                    [self.x_index, oor_index(self.y_index + self.i, self.yl),
                                     oor_index(self.z_index - self.i, self.zl)],
                                    [self.x_index, oor_index(self.y_index + self.i, self.yl),
                                     oor_index(self.z_index + self.i, self.zl)]])

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
        self.nnn_vector = np.array([[self.x_index, oor_index(self.y_index - 1, self.yl), self.z_index],
                                    [self.x_index, oor_index(self.y_index + 1, self.yl), self.z_index],
                                    [oor_index(self.x_index + self.i2, self.xl), oor_index(self.y_index + 1, self.yl),
                                     self.z_index],
                                    [oor_index(self.x_index + self.i2, self.xl), self.y_index - 1, self.z_index],
                                    [oor_index(self.x_index + self.i2, self.xl), self.y_index,
                                     oor_index(self.z_index + 2, self.zl)],
                                    [self.x_index, self.y_index, oor_index(self.z_index + 2, self.zl)],
                                    [self.x_index, oor_index(self.y_index + 1, self.yl),
                                     oor_index(self.z_index + 2, self.zl)],
                                    [self.x_index, self.y_index - 1, oor_index(self.z_index + 2, self.zl)],
                                    [oor_index(self.x_index + self.i2, self.xl), self.y_index, self.z_index - 2],
                                    [self.x_index, self.y_index, self.z_index - 2],
                                    [self.x_index, oor_index(self.y_index + 1, self.yl), self.z_index - 2],
                                    [self.x_index, self.y_index - 1, self.z_index - 2]])

        self.c = 0  # The occupation number, c = 1 is a Lithium atom c = 0 is a vacant site.
        self.temp_neighbours = []  # Temporarily stores the neighbours before converting to a numpy array.

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

        print("x index: ", self.x_index, "y index: ", self.y_index, "z index: ", self.z_index)
        print("x pos: ", self.position[0], "y pos: ", self.position[1], "z pos: ", self.position[2])
        print()

    def get_nn(self, grid):
        self.temp_neighbours = []

        # Try and except block used in case there is an index referenced out of range then the program doesnt quit.
        try:
            for vector in self.nn_vector:
                self.temp_neighbours.append(grid[vector[0]][vector[1]][vector[2]])
        except Exception as e:
            print(e)
            print("Error occurred at index: ", self.x_index, self.y_index, self.z_index)

        self.neighbours = np.array(self.temp_neighbours)

    def get_nnn(self, grid):
        self.temp_neighbours = []
        try:
            for vector in self.nnn_vector:
                self.temp_neighbours.append(grid[vector[0]][vector[1]][vector[2]])
        except Exception as e:
            # This will occur when the index is out of range od the grid.
            print(e)

        self.nnn = np.array(self.temp_neighbours)

    def node_hamiltonian(self, mu):
        sum_of_nn = 0
        sum_of_nnn = 0
        nn_term = 0
        nnn_term = 0

        for nn in self.neighbours:
            sum_of_nn += (self.c * nn.c)
        nn_term = sum_of_nn * self.j1

        for nnn in self.nnn:
            sum_of_nnn += (self.c * nnn.c)
        nnn_term = sum_of_nnn * self.j2

        self.chemical_potential_term = - self.c * (self.eps + mu)

        return nn_term + nnn_term + self.chemical_potential_term  # This is the hamiltonian of 1 atom


def monte_carlo(grid, grid_length, kb, T, mu):
    """
    This method contains the metropolis monte carlo algorithm and is classed as 1 monte carlo step.

    :param grid: 3d numpy array containing the objects Atom.
    :param grid_length: Number of unit cells per side
    :param kb: Boltzamnn constant
    :param T: Temperature in K
    :param mu: Chemical potential
    :return: The new grid with 1 or 0 changes.
    """
    rand_atom = grid[random.randint(0, grid_length - 1)][random.randint(0, (grid_length * 2) - 1)][
        random.randint(0, (grid_length * 4) - 1)]

    current_h = rand_atom.node_hamiltonian(mu)
    rand_atom.swap_c()
    new_h = rand_atom.node_hamiltonian(mu)
    delta_h = current_h - new_h  # could be the other way around?
    if delta_h < 0:
        # Only keep the swap if a random float between 0 and 1 is <= P
        rand_p = random.random()  # Generates a random number between 0 and 1
        p = math.exp(-abs(delta_h) / (kb * T))
        if rand_p > p:
            # Don't accept change and reverse swap. So overall no swap is made
            rand_atom.swap_c()

    return grid


def get_grid(grid_length):
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
                grid[x][y][z] = Atom(x, y, z, grid_length)

    # Initialise the neighbours for all atoms
    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                grid[x][y][z].get_nn(grid)
                grid[x][y][z].get_nnn(grid)

    return grid


def oor_index(number, axis_length):
    """
    Handles 'Out Of Range' indexes in the grid and applies the boundary conditions

    :param number: This will be the index of a position that may be greater than the size of the grid
    :param axis_length: This is the axis length for a given axis. z axis length = 2 * y axis length = 4 * x axis length.
    :return: If index is out of range return the modulus.
    """

    if number >= axis_length:
        return number % axis_length  # New index that loops back to the start of the array
    else:
        return number  # Make no changes


def lattice_hamiltonian(grid, grid_length, eps, mu):
    """
    Calculates the total hamiltonian

    :param grid: 3D numpy array containing the object atom.
    :param grid_length: Number of unit cells
    :param eps: Constant
    :param mu: Chemical potential
    :return: hamiltonian of the whole grid
    """

    hamiltonian = 0
    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                hamiltonian += (grid[x][y][z].node_hamiltonian() - (eps + mu) * grid[x][y][z].c)

    return hamiltonian


def get_internal_energy(grid, grid_length, eps, mu):
    """
        Calculates the total internal energy

        :param grid: 3D numpy array containing the object atom.
        :param grid_length: Number of unit cells
        :param eps: Constant
        :return: internal energy (U) of the whole grid
        """

    internal_energy = 0
    occupancy = 0
    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                internal_energy += (grid[x][y][z].node_hamiltonian(mu) - grid[x][y][z].chemical_potential_term - eps *
                                    grid[x][y][z].c)
                occupancy += grid[x][y][z].c

    return internal_energy, occupancy


def thermal_fluctuations(grid, grid_length, kb, T, mu, eps):
    sample_frequency = 20
    total_iterations = 1000
    total_rows = total_iterations / sample_frequency
    data_array = np.zeros((int(total_rows), 4))  # Columns for U, N, UN and N^2.
    row_count = 0

    for i in range(total_iterations):
        monte_carlo(grid, grid_length, kb, T, mu)
        if i % sample_frequency == 0:
            internal_energy, occupancy = get_internal_energy(grid, grid_length, eps, mu)
            internal_energy_times_occupancy = internal_energy * occupancy
            occupancy_squared = occupancy ** 2
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
    varience = mean_nn - mean_n ** 2
    cov_un = mean_un - (mean_u * mean_n)

    return varience, cov_un


def get_mole_fraction(grid, grid_length):
    """
    Returns the mole fraction of lithium atoms in each sub lattice.
    """
    counter = [[0, 0], [0, 0]]

    for z in range(grid_length * 4):
        for y in range(grid_length * 2):
            for x in range(grid_length):
                if z % 2 == 0:
                    counter[0][0] += grid[x][y][z].c
                    counter[0][1] += 1
                else:
                    counter[1][0] += grid[x][y][z].c
                    counter[1][1] += 1

    total_num_of_li = counter[1][0] + counter[0][0]
    total_num_of_sites = counter[1][1] + counter[0][1]
    total_mole_fraction = total_num_of_li / total_num_of_sites
    print("Li mole fraction in sub lattice 1:", counter[0][0] / counter[0][1], " Li mole fraction in sub lattice 2: ",
          counter[1][0] / counter[1][1])
    print("Total Li atoms:", total_num_of_li, " Total sites:", total_num_of_sites, " Total mole fraction:",
          total_mole_fraction)

    return total_mole_fraction, counter[0][0] / counter[0][1], counter[1][0] / counter[1][1]


if __name__ == '__main__':
    grid_length = 10  # This is the number of unit cells - contains 8 lithium atoms.
    mu = -3.9  # Chemical potential in eV    -4.30 < mu < -3.88
    T = 298  # Temperature in K
    kb = 8.617e-5  # Boltzmann constant in eV/K
    eps = 4.12  # in eV

    grid = get_grid(grid_length)  # Returns a numpy 3d array [x][y][z] of each primitive cell.

    number_of_mcs = 50000
    start_mu = -4.3  # Start chemical potential
    end_mu = - 3.88
    step_size_mu = 0.01  # Step size for chemical potential
    rows = math.ceil((end_mu - start_mu) / step_size_mu)
    row_count = 0
    results_array = np.zeros((rows + 1, 5))  # Columns for delta S, x, mu, sl1 mole fraction, sl2 mole fraction

    for chem in np.arange(start_mu, end_mu + step_size_mu, step_size_mu):
        print("Chemical potential: ", chem)
        for i in range(number_of_mcs):
            grid = monte_carlo(grid, grid_length, kb, T, chem)

        total_mole_fraction, sl1_mole_fraction, sl2_mole_fraction = get_mole_fraction(grid,
                                                                                      grid_length)  # Prints the final mole fraction

        var, cov = thermal_fluctuations(grid, grid_length, kb, T, chem,
                                        eps)  # Prints the data frame for the values of U and N

        delta_entropy = (1 / T) * (cov / var - chem)
        results_array[row_count, 0] = delta_entropy
        results_array[row_count, 1] = total_mole_fraction
        results_array[row_count, 2] = chem
        results_array[row_count, 3] = sl1_mole_fraction
        results_array[row_count, 4] = sl1_mole_fraction
        row_count += 1
        print("----------")

    results_dataframe = pd.DataFrame(data=results_array, columns=["Delta Entropy", "Mole Fraction of Li",
                                                                  "Chemical potential", "Mole fraction sub lattice 1",
                                                                  "Mole fraction sub lattice 2"])

    cwd = os.getcwd()
    path = cwd + "/monte_carlo_results.csv"
    results_dataframe.to_csv(path, index=False)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    results_dataframe.plot(kind='scatter', ax=axes[0, 0], x='Mole Fraction of Li', y='Chemical potential', color='red')
    results_dataframe.plot(kind='scatter', ax=axes[0, 1], x='Mole Fraction of Li', y='Delta Entropy', color='red')
    ax1 = results_dataframe.plot(kind='scatter', ax=axes[1, 0], x='Mole Fraction of Li',
                                 y='Mole fraction sub lattice 1', color='red')
    results_dataframe.plot(kind='scatter', ax=ax1, x='Mole Fraction of Li', y='Mole fraction sub lattice 2',
                           color='blue')

    plt.show()
