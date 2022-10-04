#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Visualizer():
    def __init__(self, nuclear_number, nuclear_coordinate):
        self._nuclear_number = nuclear_number
        self._nuclear_coordinate = nuclear_coordinate

    @staticmethod
    def conv_log_scale(values):
        thres = 1.e-6

        # All zero values
        if np.amax(values) == 0.0 and np.amin(values) == 0.0:
            return values

        else:
            values = values / thres
            for i in range(len(values)):
                for j in range(len(values)):
                    if np.abs(values[i, j]) < 1.0:
                        values[i, j] = 0.0
                    else:
                        values[i, j] = np.sign(values[i, j]) * np.log10(np.abs(values[i, j]))

        return values

    def contour_map(self, grids, values, pic_name, x_range, y_range, target, xy_index, dim_map="2d"):

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.rcParams["xtick.major.size"] = 8
        plt.rcParams["ytick.major.size"] = 8

        if dim_map.lower() == "2d":
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for atomidx, atom in enumerate(target):
                # H
                if atom == 1:
                    atom_raidus = 0.25
                    atom_color = "white"
                # He
                elif atom == 2:
                    atom_raidus = 0.31
                    atom_color = "white"
                # Be
                elif atom == 4:
                    atom_raidus = 1.05
                    atom_color = "white"
                # B
                elif atom == 5:
                    atom_radius = 0.85
                    atom_color = "pink"
                # C
                elif atom == 6:
                    atom_radius = 0.70
                    atom_color = "gray"
                # N
                elif atom == 7:
                    atom_radius = 0.65
                    atom_color = "blue"
                # O
                elif atom == 8:
                    atom_radius = 0.60
                    atom_color = "red"
                # F
                elif atom == 9:
                    atom_radius = 0.50
                    atom_color = "white"
                # ne
                elif atom == 10:
                    atom_radius = 0.38
                    atom_color = "white"
                else:
                    raise NotImplementedError(
                        "Atom number %s cannot be treated in the cube generation." % (str(atom)))

                atom_radius = 0.2
                atom = patches.Circle(
                    xy=[self._nuclear_coordinate[atomidx, xy_index[0]], self._nuclear_coordinate[atomidx, xy_index[1]]], radius=atom_radius * 0.6, fc=atom_color, ec='black', alpha=1.0)
                ax.add_patch(atom)

            ax.set_xticks(x_range)
            ax.set_yticks(y_range)
            ax.set_xticklabels(x_range, fontsize=16, fontname='Arial', position=(0.0, -0.02))
            ax.set_yticklabels(y_range, fontsize=16, fontname='Arial', position=(-0.02, 0.0))
            xticklabels = ax.get_xticklabels()
            yticklabels = ax.get_yticklabels()
            xlabel = "a"
            ylabel = "b"
            ax.set_xlabel("$\it{x}$ / Å", fontsize=18, fontname='Arial')
            ax.set_ylabel("$\it{y}$ / Å", fontsize=18, fontname='Arial')

            min_value = np.amin(values)
            max_value = np.amax(values)

            # For the reference molecule
            if min_value == 0.0 and max_value == 0.0:
                # To avoid the error
                contour_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                map_color = 'Blues'
            else:
                if min_value >= 0.0:
                    contour_range = np.linspace(0.0, 7.0, 40)
                    map_color = 'Blues'
                else:
                    contour_range = np.linspace(-7.0, 7.0, 40)
                    map_color = 'RdBu_r'

            values = self.conv_log_scale(values)

            # pcm = ax.pcolormesh(grids[0], grids[1], values,
            #            norm=colors.SymLogNorm(linthresh=0.01, linscale=0.000001,
            #                                   vmin=0.0, vmax=1.0),
            #            cmap='Blues')
            # fig.colorbar(pcm, ax=ax)

            # pcm = ax.pcolormesh(grids[0], grids[1], values, cmap='Blues', vmin=np.min(values))
            # fig.colorbar(pcm, ax=ax)

            ax = plt.contour(grids[0], grids[1], values, contour_range, colors='black')

            ax = plt.contourf(grids[0], grids[1], values, contour_range, cmap=map_color)
            # ax = plt.colorbar(ticks=contour_range, label="contour level", format='%1.3f')

            plt.xlim(min(x_range), max(x_range))
            plt.ylim(min(y_range), max(y_range))


            plt.gca().set_aspect('equal')
            fig.tight_layout()
            plt.savefig("%s.pdf" % str(pic_name), format='pdf', dpi=900)

            plt.close()
            plt.clf

        else:
            raise NotImplementedError("only 2D contour map is available.")
