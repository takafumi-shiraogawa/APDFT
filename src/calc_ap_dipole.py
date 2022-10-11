import numpy as np
import csv

# Because it is difficult to import APDFT_Proc of apdft_interface.py of Lime,

def get_property_values(property_name, dict, num_mol, apdft_order = 1):
  """ Read property values """
  property_values = np.zeros(num_mol)
  for i, row in enumerate(dict):
    property_values[i] = np.array(
        row["%s%s%s" % (str(property_name), "_order", str(apdft_order))], dtype=np.float64)

  return property_values

def read_potential_energies(num_mol, apdft_order, path_potential_energies):
    """ Read potential energy of target molecules
    Args:
      path_potential_energies  : A string of path of APDFT potential energies, e.g., /home/test/energies.csv
    Returns:
      potential_energies       : A (the number of molecules) array of potential energies of target molecules. [Hartree]
    """
    file_total_energies = open(path_potential_energies, "r")
    dict_total_energies = csv.DictReader(file_total_energies)
    potential_energies = get_property_values("total_energy", dict_total_energies, num_mol, apdft_order)
    file_total_energies.close()

    return potential_energies

def read_ele_dipoles(num_mol, apdft_order, path_ele_dipoles):
    """ Read atomic forces of target molecules
    Args:
      path_atomic_forces  : A string of path of APDFT atomic forces, e.g., /home/test/ver_atomic_forces.csv
    Returns:
      atomic_forces       : A (the number of molecules, the number of atoms, 3) array of atomic forces of target molecules. [Hartree / Bohr]
    """
    ele_dipole_moments = np.zeros((num_mol, 3))
    for didx, dim in enumerate('xyz'):
      file_ele_dipole_moments = open(path_ele_dipoles, "r")
      dict_ele_dipole_moments = csv.DictReader(file_ele_dipole_moments)
      ele_dipole_moments[:, didx] = get_property_values(
          "nuc_dipole_moment_%s" % (str(dim)), dict_ele_dipole_moments, num_mol, apdft_order)
      file_ele_dipole_moments.close()

    return ele_dipole_moments

au_to_debye = 2.54174776

# TODO: automation
num_mol = 3
max_apdft_order = 2

max_apdft_order += 1

field_direc = []
field_direc.append('./finite_difference/x_+/')
field_direc.append('./finite_difference/x_-/')
field_direc.append('./finite_difference/y_+/')
field_direc.append('./finite_difference/y_-/')
field_direc.append('./finite_difference/z_+/')
field_direc.append('./finite_difference/z_-/')

energies = np.zeros((len(field_direc), num_mol, max_apdft_order))

# Get total energies
for i, path in enumerate(field_direc):
  for j in range(max_apdft_order):
    energies[i, :, j] = read_potential_energies(num_mol, j, "%s/energies.csv" % path)

# Get nuclear dipoles
nuc_dipoles = read_ele_dipoles(num_mol, 0, "%s/nuc_dipoles.csv" % "./finite_difference/x_+")

ele_dipoles = np.zeros((num_mol, max_apdft_order, 3))
total_dipoles = np.zeros((num_mol, max_apdft_order, 3))

# Calculate electronic dipoles
for i in range(num_mol):
  for j in range(max_apdft_order):
    for k in range(3):
      ele_dipoles[i, j, k] = (energies[2 * k, i, j] - energies[2 * k + 1, i, j]) / (2.0 * 0.01)

with open("ele_dipole.out", mode='w') as f_out:
  for i in range(num_mol):
    print("Molecule: ", i, file=f_out)
    print("x, y, z", file=f_out)
    for j in range(max_apdft_order):
      print(*ele_dipoles[i, j, :], file=f_out)
      if j == max_apdft_order - 1:
        print("", file=f_out)

# Add nuclear term
for i in range(num_mol):
  for j in range(3):
    ele_dipoles[i, :, j] += nuc_dipoles[i, j]

with open("total_dipole.out", mode='w') as f_out:
  for i in range(num_mol):
    print("Molecule: ", i, file=f_out)
    print("x, y, z", file=f_out)
    for j in range(max_apdft_order):
      print(*ele_dipoles[i, j, :], file=f_out)
      if j == max_apdft_order - 1:
        print("", file=f_out)