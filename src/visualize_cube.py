import os
import sys
import shutil
import numpy as np
import apdft
from apdft import pyscf_interface
from apdft import visualizer

#: Conversion factor from Angstrom to Bohr
angstrom = 1 / 0.52917721067

nuclear_numbers, coordinates = apdft.read_xyz("mol.xyz")

# Input
# apdft/pyscf_interface.py should be changed!!!
# N2 or benzene
target_mol = 'n2'
# target_mol = 'benzene'

# Get the calculated density maps (cubes)
# For N2
if target_mol == 'n2':
    div_elements = 251
# For benzene
elif target_mol == 'benzene':
    div_elements = 301

# div_elements is the number of divided spatial elements per axis.
# These values should be match with the setting in pyscf_interface.py.
cube_density_coords = np.zeros((div_elements ** 3, 3))
cube_density_values = np.zeros((div_elements, div_elements, div_elements))
if len(sys.argv) == 1:
    pyscf_mol = pyscf_interface.PySCF_Mol(
        nuclear_numbers, coordinates, div_elements)
    cube_density_coords[:, :], cube_density_values[:, :, :] = pyscf_mol.read_cube('.')

pyscf_mol = pyscf_interface.PySCF_Mol(nuclear_numbers, coordinates, div_elements)

# Input
# For N2
if target_mol == 'n2':
    xy_index = [2, 0]
# For benzene
elif target_mol == 'benzene':
    xy_index = [0, 1]

# Plot 2D counter maps of the densities
name_pic_2d_map = "%s" % ("density2Dmap")
density_2d_map = visualizer.Visualizer(nuclear_numbers, coordinates)
test_xy_coords_densities = np.zeros((2, div_elements))
# For x axis
# angstrom converts Angstrom to Bohr
test_xy_coords_densities[0] = np.unique(cube_density_coords[:, xy_index[0]]) / angstrom
# For y axis
test_xy_coords_densities[1] = np.unique(cube_density_coords[:, xy_index[1]]) / angstrom

# For N2
if target_mol == 'n2':
    x_range = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    y_range = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
# For benzene
elif target_mol == 'benzene':
    x_range = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    y_range = x_range

# Perturbed electron density
if len(sys.argv) == 1:
    # For N2
    if target_mol == 'n2':
        density_2d_map.contour_map(test_xy_coords_densities, cube_density_values[:, int(
            (div_elements - 1) / 2), :], name_pic_2d_map, x_range, y_range, nuclear_numbers, xy_index)
    # For benzene
    elif target_mol == 'benzene':
        density_2d_map.contour_map(test_xy_coords_densities, np.transpose(cube_density_values[:, :, int(
            (div_elements - 1) / 2)]), name_pic_2d_map, x_range, y_range, nuclear_numbers, xy_index)

if len(sys.argv) == 1:
    sys.exit()


### Compute density difference
# Inputs
par_var1 = sys.argv[1]
par_var2 = sys.argv[2]
par_var1 = str(par_var1)
par_var2 = str(par_var2)
os.mkdir('cube1')
os.mkdir('cube2')
shutil.copyfile(par_var1, "cube1/cubegen.cube")
shutil.copyfile(par_var2, "cube2/cubegen.cube")

# Read cubes
cube_diff_density_coords = np.zeros((2, div_elements ** 3, 3))
cube_diff_density_values = np.zeros((2, div_elements, div_elements, div_elements))
pyscf_mol = pyscf_interface.PySCF_Mol(
    nuclear_numbers, coordinates, div_elements)
cube_diff_density_coords[0, :, :], cube_diff_density_values[0,
                                                            :, :, :] = pyscf_mol.read_cube('./cube1/')
pyscf_mol = pyscf_interface.PySCF_Mol(
    nuclear_numbers, coordinates, div_elements)
cube_diff_density_coords[1, :, :], cube_diff_density_values[1,
                                                            :, :, :] = pyscf_mol.read_cube('./cube2/')

test_xy_coords_densities = np.zeros((2, div_elements))
# For x axis
# angstrom converts Angstrom to Bohr
test_xy_coords_densities[0] = np.unique(cube_diff_density_coords[0, :, xy_index[0]]) / angstrom
# For y axis
test_xy_coords_densities[1] = np.unique(cube_diff_density_coords[0, :, xy_index[1]]) / angstrom

pyscf_mol = pyscf_interface.PySCF_Mol(nuclear_numbers, coordinates, div_elements)

name_pic_2d_map = "%s" % ("diffdensity2Dmap")

# For N2
if target_mol == 'n2':
    density_2d_map.contour_map(test_xy_coords_densities, cube_diff_density_values[0, :, int(
        (div_elements - 1) / 2), :] - cube_diff_density_values[1, :, int((div_elements - 1) / 2), :], name_pic_2d_map, x_range, y_range, nuclear_numbers, xy_index)
# For benzene
elif target_mol == 'benzene':
    density_2d_map.contour_map(test_xy_coords_densities, np.transpose(cube_diff_density_values[0, :, :, int(
        (div_elements - 1) / 2)] - cube_diff_density_values[1, :, :, int((div_elements - 1) / 2)]), name_pic_2d_map, x_range, y_range, nuclear_numbers, xy_index)

# print(cube_diff_density_values[0, :, int((div_elements - 1) / 2), :] - cube_diff_density_values[1, :, int((div_elements - 1) / 2)])

shutil.rmtree('cube1/')
shutil.rmtree('cube2/')