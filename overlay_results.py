import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class DisplayResults:

    def __init__(self):
        pass

    def display(self):
        cwd = os.getcwd()
        path = cwd + "/results"
        df1 = pd.read_csv(path + "/Na_monte_carlo_results_tri3.csv")  # black line
        df2 = pd.read_csv(path + "/Na_monte_carlo_results_ss2.csv")  # green line
        df3 = pd.read_csv(path + "/Na_monte_carlo_results_sites2.csv")  # blue line
        dfE = pd.read_csv(path + "/experimental_data.csv")  # grey line

        points = 1000
        x_pos = np.linspace(0, 1, points)
        y_pos = np.linspace(0, 1, points)
        s_x = np.linspace(0, 1, points)
        s_y = np.linspace(0, 1, points)
        l = 0.3
        R = -0.0000862  # eV/K.Site
        T = 288  # K
        for index, x in enumerate(x_pos):
            if x < l:
                s_y[index] = R * (x * np.log(x/l) - (x-l)*np.log((l-x)/l))
                y_pos[index] = T * R * (np.log(x/l) - np.log((l-x)/l))
            else:
                s_y[index] = R * l * ((x/l - 1) * np.log(x/l - 1) + (1-x)/l * np.log((1-x)/l) - (1-l)/l * np.log((1-l)/l))
                y_pos[index] = T * R * (np.log(x/l - 1) - np.log(1/l - x/l))

        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

        ax1 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                              markersize=2, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        #df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
        #         markersize=2, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        #df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
        #         markersize=2, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        dfE.plot(linestyle='-', color='grey', lw=0.3, marker='o', markeredgecolor='grey',
                 markersize=2, ax=axes[0, 0], x='x', y='OCV')

        ax2 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                              markersize=2, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        #df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
        #         markersize=2, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        #df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
        #         markersize=2, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        dfE.plot(linestyle='-', color='grey', lw=0.3, marker='o', markeredgecolor='grey',
                 markersize=2, ax=axes[0, 1], x='x', y='Entropy dS/dx')

        #ax2.plot(x_pos, y_pos)  # Plots the ideal p.m. entropy

        #axes[1, 0].plot(s_x, s_y)  # Plots the entropy.

        ax3 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                       markersize=2, ax=axes[1, 0], x='Chemical potential', y='dq/de')
        #df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
        #         markersize=2, ax=axes[1, 0], x='Chemical potential', y='dq/de')
        #df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
        #         markersize=2, ax=axes[1, 0], x='Chemical potential', y='dq/de')
        dfE.plot(linestyle='-', color='grey', lw=0.3, marker='o', markeredgecolor='grey',
                 markersize=2, ax=axes[1, 0], x='OCV', y='dQdV')

        ax4 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                       markersize=2, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')
        #df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
        #         markersize=2, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')
        #df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
        #         markersize=2, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')
        dfE.plot(linestyle='-', color='grey', lw=0.3, marker='o', markeredgecolor='grey',
                 markersize=2, ax=axes[1, 1], x='x', y='Enthalpy dH/dx')

        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
        ax3.set_xlim([0, 1])
        ax4.set_xlim([0, 1])

        ax1.set_xlabel('Na content, x')
        ax2.set_xlabel('Na content, x')
        ax3.set_xlabel('Voltage V')
        ax4.set_xlabel('Na content, x')

        ax1.set_ylabel('Voltage [V]')
        ax2.set_ylabel('dS/dx [eV/Na site]')
        ax3.set_ylabel('dq/de')
        ax4.set_ylabel('Partial molar enthalpy [eV/Na site]')

        # fig.suptitle('black: eps1=[0.123 to -0.477]\ngreen: eps1=[-0.077 to -0.677]\nblue: eps1=[-0.277 to -0.877]')
        # ax1.legend(['Uniform distribution [0.077 to 0.677]', 'Normal distrubition, mu=0.377 sig=0.1'])

        #plt.savefig(fig_path)
        plt.show()


    def display_averaging(self):
        cwd = os.getcwd()
        path = cwd + "/results"
        df1 = pd.read_csv(path + "/average_U.csv")  # black line
        df2 = pd.read_csv(path + "/average_N.csv")  # green line
        chem = 10  # from 0 to 46

        s1 = df1.iloc[chem]
        s1.plot()
        #s2 = df2.iloc[chem]
        #s2.plot()

        plt.show()

    def test_uniform(self):
        s = np.random.uniform(-1, 0, 50000)
        count, bins, ignored = plt.hist(s, 15, density=True)
        plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        plt.show()

    def test_triangular(self):
        h = plt.hist(np.random.triangular(-1, -0.17, -0.17, 100000), bins=200,
                     density=True)
        plt.show()


if __name__ == '__main__':
    dr = DisplayResults()
    dr.display()
    #dr.display_averaging()
    #dr.test_uniform()
    #dr.test_triangular()
