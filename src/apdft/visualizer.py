#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Visualizer():
    def __init__(self, nuclear_number, nuclear_coordinate):
        self._nuclear_number = nuclear_number
        self._nuclear_coordinate = nuclear_coordinate

    def contour_map(self, grids, values, pic_name, x_range, y_range, target, xy_index, dim_map="2d"):

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.rcParams["xtick.major.size"] = 8
        plt.rcParams["ytick.major.size"] = 8

        if dim_map.lower() == "2d":
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for atomidx, atom in enumerate(target):
                if atom == 1:
                    atom_raidus = 0.53
                    atom_color = "gray"
                elif atom == 2:
                    atom_raidus = 0.31
                    atom_color = "white"
                elif atom == 4:
                    atom_raidus = 1.12
                    atom_color = "white"
                elif atom == 5:
                    atom_radius = 0.87
                    atom_color = "pink"
                elif atom == 6:
                    atom_radius = 0.67
                    atom_color = "gray"
                elif atom == 7:
                    atom_radius = 0.56
                    atom_color = "blue"
                elif atom == 8:
                    atom_radius = 0.48
                    atom_color = "red"
                elif atom == 9:
                    atom_radius = 0.42
                    atom_color = "white"
                elif atom == 10:
                    atom_radius = 0.38
                    atom_color = "white"
                else:
                    raise NotImplementedError(
                        "Atom number %s cannot be treated in the cube generation." % (str(atom)))

                atom = patches.Circle(
                    xy=[self._nuclear_coordinate[atomidx, xy_index[0]], self._nuclear_coordinate[atomidx, xy_index[1]]], radius=atom_radius * 0.75, fc=atom_color, alpha=1.0)
                ax.add_patch(atom)

            ax.set_xticks(x_range)
            ax.set_yticks(y_range)
            ax.set_xticklabels(x_range, fontsize=16, fontname='Arial', position=(0.0, -0.005))
            ax.set_yticklabels(y_range, fontsize=16, fontname='Arial', position=(-0.005, 0.0))
            xticklabels = ax.get_xticklabels()
            yticklabels = ax.get_yticklabels()
            xlabel = "a"
            ylabel = "b"
            ax.set_xlabel("$\it{x}$ / Å", fontsize=18, fontname='Arial')
            ax.set_ylabel("$\it{y}$ / Å", fontsize=18, fontname='Arial')

            ax = plt.contour(grids[0], grids[1], values, np.linspace(0.005, 1.0, 10),colors='black')
            ax = plt.contourf(grids[0], grids[1], values, np.linspace(0.005, 1.0, 10))
            ax = plt.colorbar(label="contour level", format='%1.3f')

            plt.xlim(min(x_range), max(x_range))
            plt.ylim(min(y_range), max(y_range))


            plt.gca().set_aspect('equal')
            plt.savefig("%s.pdf" % str(pic_name), format='pdf', dpi=900)

        else:
            raise NotImplementedError("only 2D contour map is available.")
