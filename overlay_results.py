import pandas as pd
import matplotlib.pyplot as plt
import os


class DisplayResults:

    def __init__(self):
        pass

    def display(self):
        df1 = pd.read_csv(r"C:\Users\samaf\OneDrive\Desktop\Simulation Projects\Na\Na_monte_carlo_results_uniform_4.csv")  # black line
        df2 = pd.read_csv(r"C:\Users\samaf\OneDrive\Desktop\Simulation Projects\Na\Na_monte_carlo_results_normal_7.csv")  # green line
        df3 = pd.read_csv(r"C:\Users\samaf\OneDrive\Desktop\Simulation Projects\Na\Na_monte_carlo_results_uniform_3,3.csv")  # blue line

        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

        ax1 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                              markersize=2, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
                 markersize=2, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')
        df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
                 markersize=2, ax=axes[0, 0], x='Total mole fraction', y='Chemical potential')

        ax2 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                              markersize=2, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
                 markersize=2, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')
        df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
                 markersize=2, ax=axes[0, 1], x='Total mole fraction', y='Partial molar entropy')

        ax3 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                              markersize=2, ax=axes[1, 0], x='Chemical potential', y='dq/de')
        df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
                              markersize=2, ax=axes[1, 0], x='Chemical potential', y='dq/de')
        df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
                              markersize=2, ax=axes[1, 0], x='Chemical potential', y='dq/de')

        ax4 = df1.plot(linestyle='-', color='black', lw=0.3, marker='o', markeredgecolor='black',
                              markersize=2, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')
        df2.plot(linestyle='-', color='green', lw=0.3, marker='o', markeredgecolor='green',
                              markersize=2, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')
        df3.plot(linestyle='-', color='blue', lw=0.3, marker='o', markeredgecolor='blue',
                 markersize=2, ax=axes[1, 1], x='Total mole fraction', y='Partial molar enthalpy')

        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
        ax3.set_xlim([0, 1])
        ax4.set_xlim([0, 1])

        ax1.set_xlabel('Na content, x')
        ax2.set_xlabel('Na content, x')
        ax3.set_xlabel('Voltage V')
        ax4.set_xlabel('Na content, x')

        ax1.set_ylabel('Voltage [V]')
        ax2.set_ylabel('dS/dx []')
        ax3.set_ylabel('dq/de')
        ax4.set_ylabel('Partial molar enthalpy []')

        fig.suptitle('black: eps1=[0.123 to -0.477]\ngreen: eps1=[-0.077 to -0.677]\nblue: eps1=[-0.277 to -0.877]')
        #ax1.legend(['Uniform distribution [0.077 to 0.677]', 'Normal distrubition, mu=0.377 sig=0.1'])

        #plt.savefig(fig_path)
        plt.show()


if __name__ == '__main__':
    dr = DisplayResults()
    dr.display()
