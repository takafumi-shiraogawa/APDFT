import numpy as np
import apdft
from apdft import pyscf_interface
from apdft import visualizer

#: Conversion factor from Angstrom to Bohr
angstrom = 1 / 0.52917721067

nuclear_numbers, coordinates = apdft.read_xyz("mol.xyz")

# Get the calculated density maps (cubes)
div_elements = 251
# div_elements is the number of divided spatial elements per axis.
# These values should be match with the setting in pyscf_interface.py.
cube_density_coords = np.zeros((div_elements ** 3, 3))
cube_density_values = np.zeros((div_elements, div_elements, div_elements))
pyscf_mol = pyscf_interface.PySCF_Mol(
    nuclear_numbers, coordinates, div_elements)
cube_density_coords[:, :], cube_density_values[:, :, :] = pyscf_mol.read_cube('.')

pyscf_mol = pyscf_interface.PySCF_Mol(nuclear_numbers, coordinates, div_elements)

# Input
xy_index = [2, 0]

# Plot 2D counter maps of the densities
name_pic_2d_map = "%s" % ("density2Dmap")
density_2d_map = visualizer.Visualizer(nuclear_numbers, coordinates)
test_xy_coords_densities = np.zeros((2, div_elements))
# For x axis
# angstrom converts Angstrom to Bohr
test_xy_coords_densities[0] = np.unique(cube_density_coords[:, xy_index[0]]) / angstrom
# For y axis
test_xy_coords_densities[1] = np.unique(cube_density_coords[:, xy_index[1]]) / angstrom

x_range = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
y_range = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

# Perturbed electron density
density_2d_map.contour_map(test_xy_coords_densities, cube_density_values[:, int(
    (div_elements - 1) / 2), :], name_pic_2d_map, x_range, y_range, nuclear_numbers, xy_index)
