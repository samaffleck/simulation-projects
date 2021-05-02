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
        self.parser.add_argument('--sites', type=int, metavar='', help='Total number of sites including nanopores and interlayers', default=1000)
        self.parser.add_argument('--l', type=float, metavar='', help='Fraction of total sites in the interlayers', default=0.3)
        self.parser.add_argument('--mcs', type=int, metavar='', help='Number of monte carlo steps', default=1000000)
        self.parser.add_argument('--eps1', type=float, metavar='', help='Point interaction term for interlayers in eV', default=-0.377)
        self.parser.add_argument('--delE', type=float, metavar='', help='Point term for nanopores - priori heterogeneity in eV', default=-0.338)
        self.parser.add_argument('--g2', type=float, metavar='', help='g2 term in eV', default=-0.043)
        self.parser.add_argument('--g3', type=float, metavar='', help='g3 term in eV', default=0.31)
        self.parser.add_argument('--a', type=float, metavar='', help='a term', default=0.0)
        self.parser.add_argument('--b', type=float, metavar='', help='b term', default=1.5)
        self.parser.add_argument('--c', type=float, metavar='', help='c term', default=1.0)

        self.args = self.parser.parse_args()

        self.eps1 = self.args.eps1
        self.eps2 = self.args.eps1 - self.args.delE

        self.lattice1_sites = int(self.args.sites * self.args.l)  # Number of sites in the interlayers.
        self.lattice2_sites = self.args.sites - self.lattice1_sites  # Number of sites in the nanopores.
        self.lattice1 = np.zeros(self.lattice1_sites)  # Array structure for interlayers where 0 is an empty site and 1 is a filled Na site.
        self.lattice2 = np.zeros(self.lattice2_sites)  # Array structure for nanopores.

    def calculate_h(self):
        """
        Calculates the hamiltonian.
        """
        pass


if __name__ == '__main__':

    mc = MonteCarlo()
