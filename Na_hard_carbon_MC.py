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
        self.parser.add_argument('--g2', type=float, metavar='', help='g2 term', default=-0.043)
        self.parser.add_argument('--g3', type=float, metavar='', help='g3 term', default=0.31)
        self.parser.add_argument('--a', type=float, metavar='', help='a term', default=0.0)
        self.parser.add_argument('--b', type=float, metavar='', help='b term', default=1.5)
        self.parser.add_argument('--c', type=float, metavar='', help='c term', default=1.0)
        self.parser.add_argument('--sample_frequency', type=int, metavar='', help='Number of mcs before a sample is taken', default=200)

        self.args = self.parser.parse_args()

        self.eps1 = self.args.eps1
        self.eps2 = self.args.eps1 - self.args.delE

        self.lattice1_sites = int(self.args.sites * self.args.l)  # Number of sites in the interlayers.
        self.lattice2_sites = self.args.sites - self.lattice1_sites  # Number of sites in the nanopores.
        self.lattice1 = np.zeros(self.lattice1_sites)  # Interlayers where 0 is an empty site and 1 is a filled Na site.
        self.lattice2 = np.zeros(self.lattice2_sites)  # Nanopores.

        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.T = 298  # Temperature in K

    def calculate_h(self, mu):
        """
        Calculates the hamiltonian.
        """
        N1 = np.sum(self.lattice1)
        N2 = np.sum(self.lattice2)
        n1 = N1/self.lattice1_sites
        # n2 = N2/self.lattice2_sites
        # m1 = self.lattice1_sites
        # m2 = self.lattice2_sites
        eps1_prime = self.eps1 + self.args.a * math.exp(-self.args.b * n1 ** self.args.c)

        #print("ham term:", eps1_prime * N1 + self.eps2 * N2, " Chem term:", (N1 + N2) * mu)
        h = eps1_prime * N1 + self.eps2 * N2 + (N1 + N2) * mu
        #print("h:", h, " eps1:", self.eps1, " eps1_prime:", eps1_prime, " N1:", N1, " N2:", N2)

        return h

    def monte_carlo(self, mu):
        rand_pos = random.randint(0, self.args.sites - 1)
        if rand_pos < self.lattice1_sites:
            lattice = self.lattice1
            index = rand_pos
            site = lattice[index]
        else:
            lattice = self.lattice2
            index = rand_pos - self.lattice1_sites
            site = lattice[index]

        current_h = self.calculate_h(mu)

        # Perform a swap
        if site == 0:
            lattice[index] = 1
        else:
            lattice[index] = 0

        new_h = self.calculate_h(mu)
        delta_h = new_h - current_h

        if delta_h >= 0:
            rand_p = random.random()
            p = math.exp(-delta_h / (self.kb * self.T))
            if rand_p > p:
                # Don't accept change and reverse swap. So overall no swap is made
                if site == 0:
                    lattice[index] = 1
                else:
                    lattice[index] = 0

    def run_simulation(self):
        mu = -4.1  # Chemical potential eV
        for i in range(self.args.mcs):
            self.monte_carlo(mu)
            if i % self.args.sample_frequency == 0:
                print("Mole fraction in lattice 1: ", np.sum(self.lattice1)/self.lattice1_sites, " Mole fraction in lattice 2: ", np.sum(self.lattice2) / self.lattice2_sites)
                print("--------")


if __name__ == '__main__':

    mc = MonteCarlo()
    mc.run_simulation()
