
from sys import argv
import argparse
import sys
import numpy as np
import random, os, csv, logging, uuid
import pickle as pkl
from time import strftime
import errno
import pylab
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt


class MonteCarlo:
    def __init__(self):  # This part of the code is run when the class is instantiated.
        self.parser = argparse.ArgumentParser(description='MC code arguments')
        self.energy_params = ['epsilon', 'j1', 'j2', 'j3', 'j4', 'j5']  # Note: we will only use 'epsilon', 'j1', 'j2'.
        self.parser.add_argument('--nneighs', type=int, help='Number of neighbours', default=3) # Never used (code does not call)
        self.parser.add_argument('--mu_ranges', type=float, help='Mu values to calculate',
                                 default=[-4.300, -3.880, 0.005],
                                 nargs='+')  # Note: the syntax is starting mu, end mu, and mu grid. Increasing mu grid means a coarser mesh of points, which reduces computational time.
        self.parser.add_argument('--epsilon', type=float, help='Point term',
                                 default=4.10) # Note this should be input in eV
        self.parser.add_argument('--j1', type=float, help='Nearest neighbour term in eV', default=0.0300) # Should be input as eV
        self.parser.add_argument('--j2', type=float, help='Next nearest neighbour term in eV', default=-0.00125)  # Should be input as eV
        self.parser.add_argument('--j3', type=float, help='Third neighbours',
                                 default=0)  # Don't use. By default this is deactivated.
        self.parser.add_argument('--n_iterations', type=int, help='Number of iterations.', default=3000000) # The number of iterations performed for each mu value
        self.parser.add_argument('--q_relaxation', type=int, help='Relaxation steps before calculating averages.',
                                 default=0) # The number of the above iterations before which averaging is not performed (i.e. used to permit equilibrium being reached)
        self.parser.add_argument('--dim', type=int, help='Number of sides of primitive cell.',
                                 default=16)  # Allows change of the size of the simulated cell
        self.parser.add_argument('--T', type=float, help='Temperature in Kelvin', default=298.0)
        self.parser.add_argument('--defect_readin', type=str, help='Directory to read in from',
                                 default=None)  # For continuing equilibrated runs, input the defect conc of the prev run.
        self.parser.add_argument('--defect_percent', type=float, help='Percentage of defects in structure. Needed for resuming serial runs',
                                 default=0.0)
        self.parser.add_argument('--T_readin', type=str, help='Temperature value read in from file', default='298.0') # For continuing equilibrated runs, input the temperature of the prev run.
        self.parser.add_argument('--number_readin', type=str, help='Number of directory to read from. Need to set manually to resume serial calculations', default=None) # Use this is lattice is already equilibrated.
        self.parser.add_argument('--parallel_dir', type=str, help='Location to save files after parallel runs. NOT NEEDED FOR SERIAL RUNS',
                                 default=None) # Unsure what this feature is
        self.parser.add_argument('--binsize', type=int, help='Frequency of averaging', default=2000) # How many MCS between each average is taken
        self.parser.add_argument('--hec', type=bool,
                                 help='Indicates if resume a serial or set of parallel runs from input file. LEAVE AS IS FOR SERIAL RUNS',
                                 default=None) # Use 'hec_s' if running on the HEC, and use None if running locally.
        self.parser.add_argument('--serial_dir', type=str, help='Number of directory to save files into. FINAL OUTPUT DIRECTORY', default=None) # Name of output directory
        self.parser.add_argument('--readin_id', type=int, help='Index of files to read in from.', default=0) # Unsure what this feature is
        self.parser.add_argument('--delta', type=float, help='J2 separation between lattices in eV', default=0.00125) # (Half of) the seperation between the J2 parameter on each of the 2 sublattices. So J2_0 = J2 - delta, J2_1 = J2 + delta
        self.parser.add_argument('-c', '--cluster', type=str, help='Whether defects are input in clusters. T/F.', default='T') # T - clustering is implemented. F - random distribution of defects.
        self.parser.add_argument('-p', '--pictures', type=int, help='How many mu values between each picture produced',
                                 default=1) # I.e pictures are produced every (blank) mu values. Set to a number above the number of mu values simulated to produce none.
        self.parser.add_argument('-A', '--Annealing_readin', type=str, help='Name of annealing file to readin', default=None) # New feature added such that annealed lattice may be read in.
        self.args = self.parser.parse_args(argv[1:])
        self.arg_dict = dict(vars(self.args))
        self.e = 1.60218E-22  # electronic charge, in keV
        self.avogadro = 6.02214086E+23  # Avogadro number
        self.boltzmann = 1.38064852e-23
        self.no_nearest = 4  # Number of nearest neighbours
        self.no_second = 12  # Number of next nearest neighbours
        self.no_third = 12  # Number of third nn.
        #        self.n_iterations = int(argv[1])    # Renamed to make more descriptive and easier to find. Previously n.
        #        self.q_relaxation = int(argv[2])    # Renamed from q, for the same reasons as n.
        #        self.dim = int(argv[3])
        self.epsilon = self.arg_dict['epsilon']
        self.j1 = self.arg_dict['j1']
        self.j2 = self.arg_dict['j2']
        self.j3 = self.arg_dict['j3']
        self.delta = self.arg_dict['delta']     # DELTA CHANGE
        self.j2_0 = self.j2 + self.delta        # DELTA CHANGE
        self.j2_1 = self.j2 - self.delta        # DELTA CHANGE
        self.binsize = self.arg_dict['binsize']
        self.dim = self.arg_dict['dim']
        self.defect_proportion = float(self.arg_dict['defect_percent']) / 100
        self.T = self.arg_dict['T']
        self.n_iterations = self.arg_dict['n_iterations']
        self.q_relaxation = self.arg_dict['q_relaxation']
        self.mu_ranges = self.arg_dict['mu_ranges']
        self.mu_min, self.mu_max, self.mu_inc = self.mu_ranges
        self.readin_id = self.arg_dict['readin_id']
        self.cluster = self.arg_dict['cluster']
        self.pictures = self.arg_dict['pictures']
        self.annealing_readin = self.arg_dict['Annealing_readin']
        self.pinned_0 = None

# ************* Paths that must be customised to your local setup
        if self.arg_dict['hec'] is not None:
            self.input_path = '/home/hpc/34/mercerm1/MC_examples/input_mc/' # Where files with coordinates of neighbours, etc can be found.
            self.output_path = '/storage/hpc/34/mercerm1/MC_examples/output_mc/' # Where output csv files, pickle files are placed. Note this is in $global_storage.
            self.visualisation_output = '/storage/hpc/34/mercerm1/MC_examples/Latticepictures/' # Where Lattice pictures are placed. Note this is in $global_storage.
        else: # Use relative paths on your local machine.
            self.input_path = 'input_mc/'
            self.output_path = 'output_mc/'
            self.visualisation_output = 'Latticepictures/'
        self.file_path_names = [self.input_path,self.output_path,self.visualisation_output]
        print(self.file_path_names)
        for fpath in self.file_path_names:
            if not os.path.exists(fpath):
                os.makedirs(fpath)

        if self.arg_dict['defect_readin'] is not None:
            self.defect_readin = self.arg_dict['defect_readin']
            self.defect_percent = float(self.defect_readin)
            self.T_readin = self.arg_dict['T_readin']
            self.number_readin = self.arg_dict['number_readin']
            self.local_readin_dir = str(self.defect_percent) + '/' + self.T_readin + '/' + str(self.number_readin) + '/'
        if self.arg_dict['parallel_dir'] is not None:
            self.parallel_dir = self.arg_dict['parallel_dir']

        self.defect_percent = self.defect_proportion * 100 # Changed to permit decimal percentages
        self.active_sites = self.dim * self.dim * self.dim * 2  # Number of Lithium sites within the lattice. 2 Li per unit cell.
        self.pinned_sites = 3 * self.active_sites * self.defect_proportion  # Each defect pins 3 Lithium sites.
        #        self.zero_energy = self.active_sites * (-self.epsilon + 0.5 * self.no_nearest * self.j1 +  0.5 * self.no_second * self.j2) # Sets energy scale (in eV) to cope with -1, +1 structure. Extra 0.5 to account for double counting of nn.

        self.file_paths() # Enhances output path such that files are saved to an organised directory structure
        self.nearest1 = np.genfromtxt(self.input_path + 'n1_1.csv', dtype=np.int, delimiter=',') # Nearest neighbour coordinates on sublattice 0
        self.nearest2 = np.genfromtxt(self.input_path + 'n1_2.csv', dtype=np.int, delimiter=',') # Nearest neighbour coordinates on sublattice 1
        self.second1 = np.genfromtxt(self.input_path + 'n2_1.csv', dtype=np.int, delimiter=',') # As above, next nearest neigbours
        self.second2 = np.genfromtxt(self.input_path + 'n2_2.csv', dtype=np.int, delimiter=',')
        self.third1 = np.genfromtxt(self.input_path + 'n3_1.csv', dtype=np.int, delimiter=',')  # we won't use the third nearest neighbour parameters.
        self.third2 = np.genfromtxt(self.input_path + 'n3_2.csv', dtype=np.int, delimiter=',')
        self.mn2 = np.genfromtxt(self.input_path + 'Neighbour_csv_files/' + 'mn2.csv', dtype=np.int, delimiter=',') # Nearest neighbour coordinates to a type 2 Mn site.
        self.mn3 = np.genfromtxt(self.input_path + 'Neighbour_csv_files/' + 'mn3.csv', dtype=np.int, delimiter=',') # Nearest neighbour coordinates to a type 3 Mn site.
        self.mn4 = np.genfromtxt(self.input_path + 'Neighbour_csv_files/' + 'mn4.csv', dtype=np.int, delimiter=',') # ...
        self.mn5 = np.genfromtxt(self.input_path + 'Neighbour_csv_files/' + 'mn5.csv', dtype=np.int, delimiter=',')
        self.lattice = np.zeros((6, self.dim, self.dim, self.dim), dtype=np.float64)  # This numpy array represents the lattice. The first dimension specifies whether each site is lithium on sublattice 0, sublattice 1, or a type of Mn site (see above). Others are spatial dimensions. Begins as array of zeroes as is unfilled.
        # self.lattice *= -0.5 # Empties the lattice and converts to spin 1/2.      # DANIEL: Have commented out
        if self.cluster == 'T':
            self.distribute_clusters()
        elif self.cluster == 'F':
            self.distribute_defects()
        # self.defect_sites = self.defect_proportion*self.active_sites  # Abs number of defects within lattice # DANIEL: Commented out line

        self.occ1 = 0 # occupation of sublattice 0 (unpinned sites only)
        self.occ2 = 0 # occupation of sublattice 1 (unpinned sites only)
        self.total_energy = 0
        self.int_energy = 0
        #        self.uint = 0
        #        self.utotal = 0
        self.sample_no = np.ceil((self.n_iterations - self.q_relaxation) / self.binsize).astype(int)  # How many averages are taken for each mu.
        if self.arg_dict['defect_readin'] is not None:
            if self.arg_dict['Annealing_readin'] is None:
                self.lattice_input_array()
            elif self.arg_dict['Annealing_readin'] is not None:
                self.annealing_input_array()
        self.headers = ['Chemical Potential (eV)', 'Electrode Potential (V)', 'Sublattice 1', 'Sub1sd', 'Sublattice 2',
                        'Sub2sd',
                        'N', 'N_sd', 'NN', 'NN_sd', 'U', 'U_sd', 'UN', 'UN_sd', 'UIE', 'UIE_sd', 'covUN', 'varNN',
                        'dU', 'dS', 'dx_dE', 'WC1', 'WC2_0', 'WC2_1', 'WC2_t'] # All the output data to the csv file.
        self.current_time = strftime("%c")
        self.file_header = 'Run on : ' + str(self.current_time) + '\n' + \
                           'No. of iterations : ' + str(self.n_iterations) + '\n' + \
                           'Relaxation steps : ' + str(self.q_relaxation) + '\n' + \
                           'No. of defects : ' + str(self.defect_proportion) + '\n' + \
                           'Number of averages : ' + str(self.sample_no) + '\n' + \
                           'Temperature : ' + str(self.T) + '\n\n'

        self.input_variables = [self.current_time, self.n_iterations, self.dim, self.q_relaxation,
                                self.defect_proportion, self.T]
        self.file_data = []  # Empty list into which output data is placed.

    def check_path_exists(self, path): # If the path does not exist, this function creates it.
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def file_paths(self):
        # Assigns all the directory paths, relative to local directory where the code is executed.
        self.unique_identifier = str(uuid.uuid1())[
                                 0:8]  # Puts some random characters at the end so each run is truly unique.
        self.check_path_exists(self.input_path)
        self.check_path_exists(self.output_path)

        if self.arg_dict['hec'] != 'hec_p':
            self.first_dir_level = self.output_path + 'serial' + '/'
        elif self.arg_dict['hec'] == 'hec_p':
            self.first_dir_level = self.output_path + 'parallel' + '/'

        self.defect_percent = round(self.defect_percent, 2)
        self.second_dir_level = self.first_dir_level + str(self.defect_percent) + '/'
        self.third_dir_level = self.second_dir_level + str(self.T) + '/' # Adding structure to output files: there are directories within the output directories corresponding to defect percentage and temp.

        if self.arg_dict['parallel_dir'] is None and self.arg_dict['serial_dir'] is None:
            # If serial dir is not specified, files will be saved in automatically numbered directories
            counter = 0
            proto_dir = self.third_dir_level + str(counter) + '/'
            while os.path.exists(proto_dir):
                proto_dir = self.third_dir_level + str(counter) + '/'
                counter += 1
            self.fourth_dir = proto_dir
        elif self.arg_dict['parallel_dir'] is not None:
            self.fourth_dir = self.third_dir_level + self.arg_dict['parallel_dir'] + '/'
        elif self.arg_dict['serial_dir'] is not None:
            self.fourth_dir = self.third_dir_level + self.arg_dict['serial_dir'] + '/'

        #        print 'len(argv)=', len(argv), 'argv[12]=', argv[12]
        self.final_dir = self.fourth_dir + self.unique_identifier + '/'
        self.output_filename = self.final_dir + 'defect_%d_temp_%d.csv' % (self.defect_percent, self.T)
        self.output_file_path = self.final_dir
        self.check_path_exists(self.final_dir) # If the path does not exist, this function creates it.
        print(self.output_file_path)

    def lattice_input_array(self): # Finds the input lattice files to be used when continuing from an equilibrated lattice.
        self.readin_dir = self.output_path + 'serial/' + self.local_readin_dir
        unique_id = os.listdir(self.readin_dir)[self.readin_id]
        print('unique_id=' + unique_id)
        self.full_readin = self.readin_dir + unique_id + '/'
        print('self.full_readin=' + self.full_readin)
        file_list = os.listdir(self.full_readin)
        lattice_input_dict = {}
        for f in file_list:
            if f.startswith('lattice'):
                mu_value = f.split('_')[-1]
                lattice_input_dict[mu_value] = f
        self.locator_dict = dict((v, k) for k, v in lattice_input_dict.items())
        self.master_dict = {}
        for filename in self.locator_dict:
            file_path = self.full_readin + filename
            mu_value = float(filename.split('_')[-1])
            try:
                with open(file_path, 'rb') as f:
                    lattice_array = pkl.load(f)
            except:
                raise
            self.master_dict[mu_value] = lattice_array

    def annealing_input_array(self):    # If wanting to resume from an annealed lattice, this function finds the lattice file to use.
        self.readin_dir = self.output_path + 'serial/' + str(self.defect_percent) + '/-4.3/' + self.annealing_readin
        unique_id = os.listdir(self.readin_dir)[self.readin_id]
        print('unique_id=' + unique_id)
        self.full_readin = self.readin_dir + '/' + unique_id + '/'
        print('self.full_readin=' + self.full_readin)
        file = 'lattice_defect_' + str(int(self.defect_percent)) + '_T_300.000'
        lattice_input = self.full_readin + file
        with open(lattice_input, 'rb') as f:
            self.annealed_lattice = pkl.load(f)


    def distribute_defects(self):
        self.pinned_total = 0 # Running tally of total of pinned sites
        self.pinned_1 = 0 # Running tally of pinned sites on sublattice 1
        self.pinned_0 = 0 # ...

        while self.pinned_total < self.pinned_sites: # So distributes pinned_sites pinned sites.
            i = random.randint(0, 1)
            j = random.randint(0, (self.dim - 1))
            k = random.randint(0, (self.dim - 1))
            l = random.randint(0, (self.dim - 1)) #Choosing random coordinates

            if self.lattice[i, j, k, l] == 0:  # Only assigns defects to the empty Li sites
                self.lattice[i, j, k, l] = 3  # Assigns the fixed Li sites with value of 3: the 'defect identity'
                if i == 0:
                    self.pinned_0 += 1
                elif i == 1:
                    self.pinned_1 += 1
                self.pinned_total += 1

    def pinnedcount(self): # Used to count the population of pinned sites, even if the lattice file is read in (usually the distribute defects performs the counting)
        self.pinned_0 = 0
        self.pinned_1 = 0
        for i in [0, 1]:
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        if self.lattice[i, j, k, l] == 3:
                            if i == 0:
                                self.pinned_0 += 1
                            elif i == 1:
                                self.pinned_1 += 1

    def distribute_clusters(self):
        self.pinned_total = 0
        self.pinned_0 = 0
        self.pinned_1 = 0

        while self.pinned_total < (self.pinned_sites):
            i = random.randint(2, 5)  # Generates a random Mn site
            j = random.randint(0, (self.dim - 1))
            k = random.randint(0, (self.dim - 1))
            l = random.randint(0, (self.dim - 1))

            if self.lattice[i, j, k, l] == 0: # Only assigns to unfilled sites

                counter = 1

                while counter <= 3: # So each 16d site has 3 pinned Li associated.
                    neighs = (self.mn2, self.mn3, self.mn4, self.mn5)
                    if i == 2: # Each type on Mn site has different neighbours which can be pinned.
                        location = {}
                        counter2 = 0
                        check = 0 # Introduced to make sure that an infinite loop is never reached, if for example only 2 of the sites nearest to the Mn are unpinned.
                        for element in neighs[0]:
                            location[counter2] = element # Loading the coordinates into an array
                            counter2 += 1
                            check += self.lattice[element[0], (element[1] + j) % self.dim, (element[2] + k) % self.dim, (element[3] + l) % self.dim]

                        site = random.randint(0, 5) # Picks a random site of the 6 nearest to the defect.
                        a = location[site][0]
                        b = (location[site][1] + j) % self.dim
                        c = (location[site][2] + k) % self.dim
                        d = (location[site][3] + l) % self.dim
                    if i == 3:
                        location = {}
                        counter2 = 0
                        check = 0
                        for element in neighs[1]:
                            location[counter2] = element
                            counter2 += 1
                            check += self.lattice[
                                element[0], (element[1] + j) % self.dim, (element[2] + k) % self.dim, (
                                            element[3] + l) % self.dim]

                        site = random.randint(0, 5)
                        a = location[site][0]
                        b = (location[site][1] + j) % self.dim
                        c = (location[site][2] + k) % self.dim
                        d = (location[site][3] + l) % self.dim
                    if i == 4:
                        location = {}
                        counter2 = 0
                        check = 0
                        for element in neighs[2]:
                            location[counter2] = element
                            counter2 += 1
                            check += self.lattice[
                                element[0], (element[1] + j) % self.dim, (element[2] + k) % self.dim, (
                                            element[3] + l) % self.dim]

                        site = random.randint(0, 5)
                        a = location[site][0]
                        b = (location[site][1] + j) % self.dim
                        c = (location[site][2] + k) % self.dim
                        d = (location[site][3] + l) % self.dim
                    if i == 5:
                        location = {}
                        counter2 = 0
                        check = 0
                        for element in neighs[3]:
                            location[counter2] = element
                            counter2 += 1
                            check += self.lattice[
                                element[0], (element[1] + j) % self.dim, (element[2] + k) % self.dim, (
                                            element[3] + l) % self.dim]

                        site = random.randint(0, 5)
                        a = location[site][0]
                        b = (location[site][1] + j) % self.dim
                        c = (location[site][2] + k) % self.dim
                        d = (location[site][3] + l) % self.dim

                    if self.lattice[a, b, c, d] == 0 and check <= 9: #Each Mn is adjacent to 6 possible sites to be filled. To avoid infinite loop 3 must be vacant, so at most 3 can be filled already. Hence 3*3 = 9.
                        self.lattice[a, b, c, d] = 3  # Assigning pinned Li sites
                        if a == 0:
                            self.pinned_0 += 1
                        elif a == 1:
                            self.pinned_1 += 1
                        self.pinned_total += 1
                        counter += 1

                    elif check >= 9: #I nfinite loop escape.
                        counter += 3

                self.lattice[i,j,k,l] = 1 # So each defect only selected once.

    def energy_init(self, mu, mode=1):
        self.total_energy = 0
        self.int_energy = 0
        self.ulattice(mu)  # Runs function at start to get right energy scale.

    def ulattice(self, mu, mode=1): # Finds the energy of the lattice prior to any changes made by iterations at each mu value.
        # Mode = -1 will start from a filled lattice and count the empty sites.
        if mode == 1:
            self.occ1 = 0
            self.occ2 = 0
        else:
            self.occ1 = int(self.active_sites / 2)
        # Mode = -1 will start from a filled lattice and count the empty sites.

        for j in range(self.dim):
            for k in range(self.dim):
                for l in range(self.dim):
                    nnsum = 0  # Refreshes counters for each unit cell.
                    nnnsum = 0
                    for i in (0, 1): # i.e iterating over all Li sites.
                        deltaE1 = 0
                        deltaE2 = 0
                        deltaE3 = 0
                        site_occ = self.lattice[i, j, k, l]
                        if i == 0 and (site_occ == 1):  # DANIEL: Changed 0.5 to 1
                            self.occ1 += 1 * mode
                        elif i == 1 and (site_occ == 1):  # DANIEL: Changed 0.5 to 1
                            self.occ2 += 1 * mode       # Updates occupancies

                        if site_occ == 3: # Reassignment so that defect energy contributions are the same as an unpinned filled site.
                            site_occ = 1

                        if i == 0:
                            neighs = (self.nearest1, self.second1, self.third1) # Loading in nearest neighbour coordinates dependent on the sublattice.
                        else:
                            neighs = (self.nearest2, self.second2, self.third2)

                        for element in neighs[0]: # Iterates over each neighbour. If the site and neighbour are filled, this makes a contribution to the energy.
                            a = self.lattice[
                                element[0], (j + element[1]) % self.dim, (k + element[2]) % self.dim, (
                                        l + element[3]) % self.dim]
                            if a == 3:
                                a == 1
                            deltaE1 += a * site_occ * self.j1

                        for element in neighs[1]:
                            a = self.lattice[
                                element[0], (j + element[1]) % self.dim, (k + element[2]) % self.dim, (
                                        l + element[3]) % self.dim]
                            if a == 3:
                                a == 1
                            E2_inc = a * site_occ
                            if i == 0:
                                deltaE2 += E2_inc * self.j2_0
                            elif i == 1:
                                deltaE2 += E2_inc * self.j2_1   # Note that the delta term is introduced to replicate a difference in next nearest neighbour interaction on each of the 2 sublattices.

                        for element in neighs[2]:
                            a = self.lattice[
                                element[0], (j + element[1]) % self.dim, (k + element[2]) % self.dim, (
                                        l + element[3]) % self.dim]
                            if a == 3:
                                a == 1

                            deltaE3 += a * site_occ * self.j3

                        #  Calculation of the Total energy and interaction energy hamiltonian
                        #  Interaction terms have a factor of a half to account for double counting
                        self.total_energy += (deltaE1 + deltaE2 + deltaE3) / 2 - (mu + self.epsilon) * site_occ
                        self.int_energy += (deltaE1 + deltaE2 + deltaE3) / 2 - self.epsilon * site_occ
        print('Energy initialised at:', self.total_energy)

    def hamiltonian(self, a, b, c, d, mu): # Finds the proposed change in energy by the flip (occupied to unoccupied or vice versa) of a selected site.
        nnsum = 0
        nnnsum = 0  # Refreshes counters and assigns each neighbour coordinate below from csv files
        nnnnsum = 0
        site = self.lattice[a, b, c, d]  # Setting the selected site

        if a == 0:
            neighs = (self.nearest1, self.second1, self.third1) # Loading in neighbour coordinates dependent on sublattice
            J2 = self.j2_0     # Note J2 is dependent on the sublattice due to the mean field delta term.
        else:
            neighs = (self.nearest2, self.second2, self.third2)
            J2 = self.j2_1

        for element in neighs[0]:
            nnocc = (self.lattice[element[0], (b + element[1]) % self.dim,
                                  (c + element[2]) % self.dim, (d + element[3]) % self.dim])
            if nnocc != 0:
                nnsum += 1  # So if a nearest neighbour site is pinned (3) or filled (1), it contributes 1 to this counter.

        for element in neighs[1]:
            nnnocc = (self.lattice[element[0], (b + element[1]) % self.dim,
                                   (c + element[2]) % self.dim, (d + element[3]) % self.dim])
            if nnnocc != 0:
                nnnsum += 1

        for element in neighs[2]:
            nnnnocc = (self.lattice[element[0], (b + element[1]) % self.dim,
                                    (c + element[2]) % self.dim, (d + element[3]) % self.dim])
            if nnnnocc != 0:
                nnnnsum += 1

        if site == 0:
            self.trial_change = (self.j1 * nnsum + J2 * nnnsum + self.j3 * nnnnsum - (
                        self.epsilon + mu))  # Change in the total energy by filling of an unoccupied site.
            self.trial_u = (
                        self.j1 * nnsum + J2 * nnnsum + self.j3 * nnnnsum - self.epsilon)  # As above, but only internal energy
        elif site == 1:
            self.trial_change = -(self.j1 * nnsum + J2 * nnnsum + self.j3 * nnnnsum - (self.epsilon + mu))     # As above, but change due to emptying a filled site.
            self.trial_u = -(self.j1 * nnsum + J2 * nnnsum + self.j3 * nnnnsum - self.epsilon)
        elif site == 3:
            self.trial_change = 0
            self.trial_u = 0  # DANIEL: Added to make sure no energy added by pinned sites (do not flip)

#    def WarrenCowley(self): # This function is called once at the end of each mu value to calculate Warren-Cowley SRO parameters. NOTE: Split parameter calculation is not correct.
#        srcount = 0 # Counts the number of Li/vacancy pairs that are nearest neighbours
#        lrcount_sublattice0 = 0 # Counts the number of Li/vacancy pairs that are next nearest neighbours on sublattice 0
#        lrcount_sublattice1 = 0 # Counts the number of Li/vacancy pairs that are next nearest neighbours on sublattice 1
#        lrcount_total = 0 # Counts the number of Li/vacancy pairs that are next nearest neighbours
#        WC_counter = 0 # Counts the total number of simulated pairs of nearest neighbours.
#        WC2_sublattice0 = 0 # Counts the total number of simulated pairs of next nearest neighbours on sublattice 0
#        WC2_sublattice1 = 0 # Counts the total number of simulated pairs that are next nearest neighbours on sublattice 1
#        WC2_total = 0 # Counts the total number of simulated pairs that are next nearest neighbours.

#        for i in [0, 1]:
#            for j in range(self.dim):
#                for k in range(self.dim):
#                    for l in range(self.dim):  # This stack of for loops iterates through each site in the lattice
#                        if i == 0:
#                            neighs = (
#                                self.nearest1, self.second1,
#                                self.third1)  # Loading in coordinates of nearest neighbours.
#                        else:
#                            neighs = (self.nearest2, self.second2, self.third2)
#                        srocc1 = self.lattice[i, j, k, l]
#                        if srocc1 == 3:
#                            srocc1 = 1  # So srocc1 is 0 if the site is vacant, and 1 if occupied (even if pinned).
#                        for element in neighs[0]:  # Iterating over each of the nearest neighbours to the simulated site.
#                            srocc2 = self.lattice[
#                                element[0], (j + element[1]) % self.dim, (k + element[2]) % self.dim, (
#                                        l + element[3]) % self.dim]
#                            if srocc2 == 3:
#                                srocc2 = 1  # So srocc2 is 0 if the neighbour site is vacant, and 1 if occupied (even if pinned).
#                            if srocc1 + srocc2 == 1:  # Only true if the pair is Li/vacancy
#                                srcount += 1  # Counts number of Li/vacancy pairs
#                            WC_counter += 1  # Counts number of pairs iterated over.

#                        for element in neighs[1]: # Iterating over each of the next nearest neighbours to the simulated site.
#                            lrocc = self.lattice[element[0], (j + element[1]) % self.dim, (k + element[2]) % self.dim, (
#                                    l + element[3]) % self.dim]
#                            if lrocc == 3:
#                                lrocc = 1 # So lrocc is 0 if the next nearest neighbour site is vacant, and 1 if occupied (even if pinned).
#                            if i == 0: # So only counting the pairs on the 0 sublattice.
#                                if srocc1 + lrocc == 1: # Only true if the pair is Li/vacancy
#                                    lrcount_sublattice0 += 1
#                                WC2_sublattice0 += 1

#                            elif i == 1: # Only counting the pairs on the 1 sublattice.
#                                if srocc1 + lrocc == 1:
#                                    lrcount_sublattice1 += 1
#                                WC2_sublattice1 += 1

#                            if srocc1 + lrocc == 1: # Counts all the next nearest neighbour pairs.
#                                lrcount_total += 1
#                            WC2_total += 1

#        x = (self.occ1 + self.occ2 + self.pinned_sites) / self.active_sites  # Fractional occupancy of lattice
#        x0 = (self.occ1 + self.pinned_0) / (self.active_sites / 2) # Fractional occupancy of sublattice 0
#        x1 = (self.occ2 + self.pinned_1) / (self.active_sites / 2) # Fractional occupancy of sublattice 1
#        # x = (self.occ1 + self.occ2) / self.active_sites

#        SRP = srcount / WC_counter  # Fraction of simulated nearest neighbour pairs that are Li/vacancy.
#        LRP0 = lrcount_sublattice0 / WC2_sublattice0    # Fraction of simulated next nearest neighbour pairs that are Li/vacancy on sublattice 0.
#        LRP1 = lrcount_sublattice1 / WC2_sublattice1    # Fraction of simulated next nearest neighbour pairs that are Li/vacancy on sublattice 1.
#        LRPt = lrcount_total / WC2_total    # Fraction of simulated next nearest neighbour pairs that are Li/vacancy.
#        if x == 0 or 1 - x == 0: # Just introduced to avoid any ZeroDivisionErrors.
#            self.WCsr = 0
#            self.WClrt = 0
#        else:
#            self.WCsr = 1 - (SRP / (2 * x * (1 - x)))  # Denominator is probability a random pair is Li/Vacancy in a solid solution.
#            self.WClrt = 1 - (LRPt / (2 * x * (1 - x)))
#        if x0 == 0 or 1 - x0 == 0:  # Just introduced to avoid any ZeroDivisionErrors.
#            self.WClr0 = 0
#        else:
#            self.WClr0 = 1 - (LRP0 / (2 * x0 * (1 - x0)))
#        if x1 == 0 or 1 - x1 == 0:
#            self.WClr1 = 0
#        else:
#            self.WClr1 = 1 - (LRP1 / (2 * x1 * (1 - x1))) # So WCsr, WClrt, WClr0, WClr1 are the Warren Cowley parameters.


    def generate_lattice(self, mu): # This function generates the lattice pictures.

        fig = pylab.figure()
        for i in range(self.dim):
            for j in range(self.dim):
                a = self.lattice[0,i,j,0] # So picks the edge cross section of sublattice 0.
                if a == 1:
                    pylab.plot([2 * i], [2 * j], '.', color='c') # Unpinned occupied sites plotted as cyan dots.
                if a == 3:
                    pylab.plot([2 * i], [2 * j], 'x', color='c')    # Pinned sites plotted as red dots.

        for i in range(self.dim):
            for j in range(self.dim):
                b = self.lattice[1,i,j,0] # So overlays the same cross section but of sublattice 1.
                if b == 1:
                    pylab.plot([(2 * i) + 1], [(2 * j) + 1], '.', color='r') # As above, but in red to distinguish between sublattices.
                if b == 3:
                    pylab.plot([(2 * i) + 1], [(2 * j) + 1], 'x', color='r')
        axes = pylab.gca()
        axes.set_xlim([-5, 35])
        axes.set_ylim([-5, 35])
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        pylab.title('mu = ' + str(round(mu,3)) + '    Defect percentage = ' + str(self.defect_percent))
        self.check_path_exists(self.visualisation_output + str(self.defect_percent) + '/' + str(self.T) + '/' + self.arg_dict['serial_dir'])
        fig.savefig(self.visualisation_output + str(self.defect_percent) + '/' + str(self.T) + '/' + self.arg_dict['serial_dir'] + '/' + str(round(1000*mu,0)) + '_picture.png')
        plt.clf() # Clears the buffer.
            # fig.savefig(self.visualisation_output + str(self.defect_percent) + '/' + str(self.T) + '/' + self.arg_dict['serial_dir'] + '/' + str(mu) + '_picture.png')


    def monte_carlo(self, mu):
        a = random.randint(0, 1)  # Chooses either the 0 or 1 'sublattice'
        b = random.randint(0, (self.dim - 1))  # Generates random numbers for selection of random sites
        c = random.randint(0, (self.dim - 1))
        d = random.randint(0, (self.dim - 1))

        site = int(self.lattice[a, b, c, d])  # DANIEL: Got rid of *2.
        self.hamiltonian(a, b, c, d, mu)  # Calls Hamiltonian function, calculates delta H for the site

        if self.trial_change <= 0: # So change in occupancy permitted if reduces the total energy.
            if site == 1:
                site = 0
                if a == 0:
                    self.occ1 += -1 # Updating the occupancies.
                else:
                    self.occ2 += -1
            elif site == 0:
                site = 1
                if a == 0:
                    self.occ1 += 1
                else:
                    self.occ2 += 1
            self.total_energy += self.trial_change # Updating total energy sums.
            self.int_energy += self.trial_u

        else: # Change in occupancy is also permitted with a boltzmann probability if the total energy increases as a result.
            e = random.random()  # Random number generated for comparison between 0 and 1
            p = np.exp(-self.trial_change / (self.T * 8.6173325E-5))  # Probability of spin changing

            if e < p:  # Comparison of random number and probability
                if site == 1:
                    site = 0
                    if a == 0:
                        self.occ1 += -1
                    else:
                        self.occ2 += -1
                elif site == 0:
                    site = 1
                    if a == 0:
                        self.occ1 += 1
                    else:
                        self.occ2 += 1
                self.total_energy += self.trial_change
                self.int_energy += self.trial_u

        self.lattice[a, b, c, d] = site  # Actually changes the numpy lattice to reflect the changes.

    def write_file(self): # Writes the csv file containing all the output data.
        self.csv_name = self.output_file_path + self.unique_identifier + '_' + str(self.defect_percent) + '_' + str(
            self.T) + '.csv'
        print(self.csv_name)
        with open(self.csv_name, 'w') as csvfile:
            file_object = csv.writer(csvfile, delimiter=',', quotechar='"')
            file_object.writerow([self.file_header])
            file_object.writerow('')
            file_object.writerow(self.headers)
            for line in self.file_data:
                file_object.writerow(line)

    def standard_dev(self, array):
        self.avg = np.sum(array) / (self.sample_no - 1)  # Calculates mean for the data set

        self.sd = np.sqrt(np.sum((array - self.avg) ** 2)) / (
                    self.sample_no - 1)  # Calculates the standard deviation for the data set

    def thermo_averaging(self): # Performs all the thermodynamic averaging calculations.
        with open(self.output_file_path + 'status_defect_%d' % self.defect_percent, 'w') as f:
            f.write(self.file_header + '\n')
            f.write('\n')

        ''' Start of Monte Carlo loop'''
        if self.arg_dict['defect_readin'] is None or self.annealing_readin is not None:
            a = np.arange(self.mu_min, self.mu_max, self.mu_inc)
            mu_range = a # Creates the array of mu values to iterate over.
            print(mu_range)
        else:
            mu_range = sorted([float(key) for key in self.master_dict.keys()])

        mu_counter = 0 # Keeps track of how many mu values have been iterated over (for picture production).

        if self.annealing_readin is not None:
            self.lattice = self.annealed_lattice
            self.pinnedcount() # Required to count number of pinned sites on each lattice (usually done by distribute defects function.

        for mu in mu_range: # Iterates over each mu value.
            #            self.energy_init    # Iterates over a range of chemical potential values
            #            if mu > -4.1:
            #                mode = -1
            #            else:
            #                mode = 1

            #            self.ulattice()
            if self.arg_dict['defect_readin'] is not None:
                if self.annealing_readin is None:
                    self.lattice = self.master_dict[mu]
            self.energy_init(mu)
            #           self.ulattice()

#            with open((self.output_file_path + '_current_mu_defect_%d') % self.defect_percent, 'w') as f, open((self.output_file_path + 'status_defect_%d') % self.defect_percent, 'a+') as f_b:
            with open((self.output_file_path + '_current_mu_defect_%d') % self.defect_percent, 'w') as f:
                f.write('mu=%.3f' % mu)
                f.flush()

                '''Refreshes counters for thermodynamic averaging'''
                x1_arr = np.empty((self.sample_no - 1))
                x2_arr = np.empty((self.sample_no - 1))
                n_arr = np.empty((self.sample_no - 1))
                nn_arr = np.empty((self.sample_no - 1))
                uie_arr = np.empty((self.sample_no - 1))
                umcs_arr = np.empty((self.sample_no - 1))
                un_arr = np.empty((self.sample_no - 1))

                avg_no = 0

                for itt in range(0, self.n_iterations):  # For each mu value, performs n_iterations MCS.
                    self.monte_carlo(float(mu))  # Runs Monte Carlo algorithm for each chemical potential value.
                    '''Thermodynamic averaging'''
                    if itt > self.q_relaxation: # Only takes averages once equilibrium reaches.
                        if itt % self.binsize == 0:  # Point of binsize - only takes averages every binsize iterations.
                            #                        self.ulattice()       # Calculate respective occupancies for sublattice 1 and 2
                            # Set 'sublattice 1' (x1) as max occupancy and 'sublattice 2' as the min occupancy lattices
                            if self.pinned_0 is not None: # I.e. defect distribution approx symmetric between sublattices.
                                y1 = (self.occ1) / (0.5 * (self.active_sites - self.pinned_sites))
                                y2 = (self.occ2) / (0.5 * (self.active_sites - self.pinned_sites))
                            else:
                                y1 = (self.occ1) / ((0.5 * self.active_sites) - self.pinned_0)
                                y2 = (self.occ2) / ((0.5 * self.active_sites) - self.pinned_1)
                            if y1 >= y2:
                                x1 = y1  # DEFECT: Since defects do not contribute to occupancies, they should not be included in normalisation.
                                x2 = y2
                            else:
                                x1 = y2
                                x2 = y1
                            # Calculate total occupancy of the lattice
                            n = (self.occ1 + self.occ2) / (self.active_sites - self.pinned_sites) # DEFECT: Since defects do not contribute to occupancies, they should not be included in normalisation.
                            nn = n * n
                            x1_arr[avg_no] = x1
                            x2_arr[avg_no] = x2
                            n_arr[avg_no] = n
                            nn_arr[avg_no] = nn

                            uie = (float(self.int_energy) / (self.active_sites - self.pinned_sites))  # Internal energy per site
                            umcs = (float(self.total_energy) / (self.active_sites - self.pinned_sites))  # Total energy per sites
                            uie_arr[avg_no] = uie
                            umcs_arr[avg_no] = umcs
                            un = uie * n
                            un_arr[avg_no] = un
                            avg_no += 1

#                            if itt % (self.binsize * 100) == 0: # Unsure of purpose. MM: for debugging
#                                f_b.write('mu=%.3f, itt=%d x1=%d, x2=%d, Etot=%.3f, U=%.3f, un=%.3f\n' % (
#                                mu, itt, self.occ1, self.occ2, self.total_energy, self.int_energy, un))
#                                f_b.flush()

#            if mu_counter % self.pictures == 0:
#              self.generate_lattice(mu)  # Rough method should print a picture every self.pictures mu values
# MM: commented out as teh code is buggy

            if (self.arg_dict['hec'] == 'hec_s') or (self.arg_dict['hec'] == None) : # Produces pickle files for runs using input. Note if doing parallel runs these aren't necessary so are not produced.
                with open(self.output_file_path + ('lattice_defect_%d_mu_%.3f') % (self.defect_percent, mu), 'wb') as f:
                    pkl.dump(self.lattice, f)

            '''Final calculation of averages for a specific chemical potential'''
            ep = mu * (-1)  # Chemical potential converted to the electrode potential
            print('EP = ', ep, ' final avg no = ', avg_no, 'sample_no = ', self.sample_no)
            x1sd = np.std(x1_arr) # Actual averaging of all the acquired data.
            X1 = np.mean(x1_arr)
            x2sd = np.std(x2_arr)
            X2 = np.mean(x2_arr)
            Nsd = np.std(n_arr)
            N = np.mean(n_arr)
            NNsd = np.std(nn_arr)
            NN = np.mean(nn_arr)
            UIEsd = np.std(uie_arr)
            UIE = np.mean(uie_arr)
            Usd = np.std(umcs_arr)
            U = np.mean(umcs_arr)
            UNsd = np.std(un_arr)
            UN = np.mean(un_arr)

#            mc.WarrenCowley()

#            WC1 = self.WCsr   # So just makes one calculation per mu value now.
#            WC2_0 = self.WClr0
#            WC2_1 = self.WClr1
#            WC2_t = self.WClrt


            print('un_arr', un_arr)

            #Below: Calculation of all relevant thermodynamic variables from averaging.
            covUN = (UN - (UIE * N))  # Calculation of cov(UN) and Var(N) from definition
            varNN = (NN - (N * N))
            N2 = N * (self.active_sites - self.pinned_sites)  # Introduced because N actually refers to x. Same for NN below.
            NN2 = NN * ((self.active_sites - self.pinned_sites) ** 2) # DEFECT: Since defects do not contribute to occupancies, they should not be included in normalisation.
            dx_dE = NN2 - (N2 ** 2)  # Just the variance on N
            dx_dE /= (self.boltzmann * self.T) * 1/(1000*self.e) * (self.active_sites - self.pinned_sites)  #Note self.e is in keV.   # DEFECT: Since defects do not contribute to occupancies, they should not be included in normalisation.

            dU_1 = covUN / varNN
            dU_kJ = dU_1 * self.avogadro * self.e  # Conversion between eV and Kj mol-1.
            mu_kJ = mu * self.avogadro * self.e
            dS_J = (1 / self.T) * (dU_kJ - mu_kJ) * 1000  # Converts to J

            attributes = [mu, ep, X1, x1sd, X2, x2sd, N, Nsd, NN, NNsd, U, Usd, UN, UNsd, UIE, UIEsd, covUN, varNN,
                          dU_kJ, dS_J, dx_dE]
            # Where the attributes are the chemical potential value, the electrode potential value, the occupancy of
            # sublattice 1 followed by the standard deviation, the occupancy of sublattice 2 followed by the standard
            # deviation, the total occupancy of the lattice (and sd), the value of the occupancy squared (and sd). The
            # total energy of the lattice (and sd) and then the interaction energy only of the lattice (and sd). Finally
            # there is the covariance of the UN values, the variance of N and then the calculated dU/dN and dS/dN values
            # in kJ/mol. DANIEL: I have added dx/dE for voltammogram plots, and each Warren Cowley parameter.
            self.file_data.append(attributes)  # Attributes is a list of lists, will eventually be written to the file.
            mu_counter += 1     # Introduced to count mu iterations for pictures

        self.write_file()
        #  print self.lattice
        #  return attribute_dictionary


if __name__ == '__main__':

    log_filename = 'logs/logfile.out'
    if not os.path.exists('logs'):
        os.mkdir('logs')

    logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    logging.debug('Run on ' + str(strftime("%c")))

    try:
        mc = MonteCarlo()  # Instantiates the class
        mc.thermo_averaging()  # Runs the thermodynamic averaging function from the class.
    except:
        logging.exception('Exception raised on main handler')
        raise
