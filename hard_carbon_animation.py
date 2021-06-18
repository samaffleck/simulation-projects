import pygame
import random
import time
import os
import pandas as pd


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)


class Node:
    def __init__(self, type, spacing):
        self.type = type
        if type == 0:
            # Nanopore
            self.colour = BLUE
            self.on = True
            self.exist = True
            self.delx = spacing / 2 * random.random()
            self.dely = spacing / 2 * random.random()
        elif type == 1:
            # Interlayer site
            self.colour = RED
            self.delx = 0
            self.dely = 0
            if random.random() < 0.48:
                self.on = True
                self.exist = True
            else:
                self.on = False
                self.exist = False


def animation(matrix, n, time):
    # Initialise variables
    atom_size = 5  # Radius
    w = 600
    h = 600
    spacing = atom_size*4  # 1 Diameter spacing
    padding = (w - (n*atom_size*2 + (n-1)*(spacing-(2*atom_size))))/2

    pygame.init()
    game_display = pygame.display.set_mode((w, h))
    game_display.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Draw nodes
    for y in range(n):
        for x in range(n):
            node = matrix[y][x]
            colour = node.colour

            if node.on == True:
                pygame.draw.circle(game_display, colour, (padding + x*spacing + node.delx, padding + y*spacing + node.dely), atom_size)

            my_font = pygame.font.SysFont('arial', 12)

    # Print time
    my_font = pygame.font.SysFont('arial', 15)
    text_surface = my_font.render('Time: ' + str(time), False, WHITE)
    game_display.blit(text_surface, (w/2, 10))


def main():
    # Define an empty cube matrix.
    # matrix[y][x].
    n = 20
    atom_size = 5  # Radius
    w = 600
    h = 600
    spacing = atom_size * 4  # 1 Diameter spacing
    padding = (w - (n * atom_size * 2 + (n - 1) * (spacing - (2 * atom_size)))) / 2
    matrix = [[Node(0, spacing) for x in range(n)] for x in range(n)]
    for y in range(n):
        for x in range(n):
            if y % 2 == 0:
                matrix[y][x] = Node(0, spacing)  # Pore
            else:
                matrix[y][x] = Node(1, spacing)  # Interlayer

    t = 0
    time_step = 1
    index = 0

    animation(matrix, n, t)

    # Running window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            animation(matrix, n, t)
            pygame.display.update()
            time.sleep(time_step)
            #matrix = update_nodes(matrix, n, index)
            index += 1
            t += time_step

        pygame.display.update()


def update_nodes(matrix, n, index):
    matrix2 = matrix
    cwd = os.getcwd()
    path = cwd + "/results"
    df1 = pd.read_csv(path + "/Na_monte_carlo_results_exp_8.csv")  # black line


if __name__ == '__main__':
    main()
