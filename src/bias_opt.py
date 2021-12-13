#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import shutil
import jinja2 as jinja

# Algorithms
# 1. Identify unique molecules
# 2. Generate weights of the molecules


# Copy inputs of an APDFT calculation from template/ to the target directry
def copy_ingredients(target):

  # Set a target directry
  copy_directry = "work/iter-%s" % (str(target))

  # Copy imp_mod_cli1.sh
  copyfile = "template/imp_mod_cli1.sh"
  shutil.copy(copyfile, copy_directry)

  # Copy imp_mod_cli2.sh
  copyfile = "template/imp_mod_cli2.sh"
  shutil.copy(copyfile, copy_directry)

  # Copy imp_mod_cli2.sh
  copyfile = "template/apdft.conf"
  shutil.copy(copyfile, copy_directry)

  return

# generate n2.xyz or n2_mod.inp with arbitrary bond length


def gener_inputs(target1, target2, target_inp):
  with open(str(target_inp)) as fh:
    template = jinja.Template(fh.read())

  env = {}
  env["bond_length1"] = target1
  env["bond_length2"] = target2

  return template.render(**env)

def get_target_value(target, dict_inp, apdft_order):
  target_values = []
  for i, row in enumerate(dict_inp):
    target_values.append(row["%s%s" % (target, str(apdft_order))])

  return np.array(target_values).astype(np.float64)


def get_weight_property(full_property, id_unique_mol, weight):
  weight_property = 0.0
  pos = -1
  for i in range(len(full_property)):
    if i in id_unique_mol:
      pos += 1
      weight_property += full_property[i] * weight[pos]

  return weight_property

def calc_weight_energy_and_gradients(path, num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights):
  # Set information on outputs of the APDFT calculation
  inp_total_energy = open("%s/energies.csv" % path, "r")
  inp_atomic_force = open("%s/ver_atomic_forces.csv" % path, "r")

  # Open the inputs
  dict_total_energy = csv.DictReader(inp_total_energy)
  dict_atomic_force = csv.DictReader(inp_atomic_force)

  full_energies = np.zeros(num_full_mol)
  full_gradients = np.zeros((num_full_mol, num_atom))
  weight_gradients = np.zeros(num_atom)

  # Obtain results
  full_energies = get_target_value(
      "total_energy_order", dict_total_energy, apdft_order)
  for i in range(num_atom):
    full_gradients[:, i] = get_target_value(
        "ver_atomic_force_%s_order" % str(i), dict_atomic_force, apdft_order)
    inp_atomic_force.close()
    inp_atomic_force = open("%s/ver_atomic_forces.csv" % path, "r")
    dict_atomic_force = csv.DictReader(inp_atomic_force)

  inp_total_energy.close()
  inp_atomic_force.close()

  full_gradients[:, :] = -full_gradients[:, :]

  # Compute weighted properties
  weight_energy = get_weight_property(full_energies, id_unique_mol, mol_weights)
  for i in range(num_atom):
    weight_gradients[i] = get_weight_property(
        full_gradients[:, i], id_unique_mol, mol_weights)

  return weight_energy, weight_gradients

def calc_weight_dipole(path, num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights):
  # Set information on outputs of the APDFT calculation
  inp_dipole = open("%s/dipoles.csv" % path, "r")

  # Open the inputs
  dict_dipole = csv.DictReader(inp_dipole)

  full_dipoles = np.zeros(num_full_mol)

  # Obtain results
  full_dipoles = get_target_value(
      "dipole_moment_z_order", dict_dipole, apdft_order)

  # Compute weighted properties
  weight_dipole = get_weight_property(
      full_dipoles, id_unique_mol, mol_weights)

  inp_dipole.close()

  return weight_dipole

# Conduct line search
def line_search(coord, energy, gradient):
  next_coord = np.zeros(len(coord))
  for i in range(len(coord)):
    # next_coord[i] = coord[i] - (energy / gradient[i])
    next_coord[i] = coord[i] - (0.2 * gradient[i])

  return next_coord


# Begin the code
if os.path.isdir("work/"):
  shutil.rmtree("work/")

# Parameters
# intial bond length
inp_coord_atom1 = 0.0
inp_coord_atom2 = 1.4
apdft_order = 1
num_atom = 2
# maximum number of bias shifts
max_bias_shift = 100
# maximum number of optimization loops
max_geom_opt = 1000
ipsilon = 0.001

# Convertor
ang_to_bohr = 1.0 / 0.529117

# Results
# APDFT energy
energy = np.zeros(max_geom_opt)
# APDFT gradient
gradient = np.zeros((num_atom, max_geom_opt))
# Atomic coordinates
coord = np.zeros((num_atom, max_geom_opt))

coord[0, 0] = inp_coord_atom1
coord[1, 0] = inp_coord_atom2

# Obtain nuclear-nuclear repulsion energies of all molecules in chemcal space
inp_nuc_energies = open("./nuc_energies.csv", "r")
dict_inp_nuc_energies = csv.DictReader(inp_nuc_energies)
full_nuc_energies = get_target_value("nuc_energy_order", dict_inp_nuc_energies, 0)
inp_nuc_energies.close()

num_full_mol = len(full_nuc_energies)

# Obtain indexes of the unique molecules
unique_nuc_energies, id_unique_mol = np.unique(
    full_nuc_energies, return_index=True, axis=0)

del full_nuc_energies
del unique_nuc_energies
gc.collect()

# The number of unique molecules
num_unique_mol = len(id_unique_mol)

# Set initial weights
mol_weights = np.zeros(len(id_unique_mol))
mol_weights[:] = 1.0 / num_unique_mol

# For development of geometry optimizer
max_bias_shift = 1

for bias_shift_idx in range(max_bias_shift):
  for geom_opt_idx in range(max_geom_opt):

    path = "work/iter-%s" % (str(geom_opt_idx))
    os.makedirs(path)

    # Copy inputs of the APDFT calculation in the working directry
    copy_ingredients(geom_opt_idx)

    # Set *.xyz
    inputfile_ori = gener_inputs(
        coord[0, geom_opt_idx], coord[1, geom_opt_idx], "template/n2.xyz")
    inputfile_mod = gener_inputs(
        coord[0, geom_opt_idx], coord[1, geom_opt_idx], "template/n2_mod.xyz")
    with open("%s/n2.xyz" % path, "w") as inp:
      inp.write(inputfile_ori)
    with open("%s/n2_mod.xyz" % path, "w") as inp:
      inp.write(inputfile_mod)

    # Implement the APDFT calculation
    os.system("( cd %s && bash imp_mod_cli1.sh )" % path)
    os.system("( cd %s && bash commands.sh )" % path)
    os.system("( cd %s && bash imp_mod_cli2.sh )" % path)

    weight_energy, weight_gradients = calc_weight_energy_and_gradients(
        path, num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights)

    weight_dipole = calc_weight_dipole(
        path, num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights)

    energy[geom_opt_idx] = weight_energy
    gradient[:, geom_opt_idx] = weight_gradients

    print("")
    print("*** RESULTS ***")
    print("Step", geom_opt_idx)
    print("Energy", energy[geom_opt_idx])
    if geom_opt_idx > 0:
      print("Energy difference",
            energy[geom_opt_idx] - energy[geom_opt_idx - 1])
    print("Gradient", gradient[:, geom_opt_idx])
    print("Coordinates", coord[:, geom_opt_idx])
    print("Bond length", abs(coord[0, geom_opt_idx] - coord[1, geom_opt_idx]))
    print("")

    # Save results
    np.savetxt('energy_hist.csv', energy[:geom_opt_idx + 1])
    np.savetxt('grad_hist.csv', np.transpose(
        gradient[:, :geom_opt_idx + 1]), delimiter=',')
    np.savetxt('geom_hist.csv', np.transpose(
        coord[:, :geom_opt_idx + 1]), delimiter=',')
    np.savetxt('bond_hist.csv', abs(
        coord[0, :geom_opt_idx + 1] - coord[1, :geom_opt_idx + 1]), delimiter=',')


    if np.amax(abs(gradient[:, geom_opt_idx])) < ipsilon:
      break

    # Perform line search and obtain renewed coordinates
    coord[:, geom_opt_idx + 1] = line_search(coord[:, geom_opt_idx],
                                             energy[geom_opt_idx], gradient[:, geom_opt_idx] / ang_to_bohr)

# Save results
np.savetxt('energy_hist.csv', energy[:geom_opt_idx + 1])
np.savetxt('grad_hist.csv', np.transpose(
    gradient[:, :geom_opt_idx + 1]), delimiter=',')
np.savetxt('geom_hist.csv', np.transpose(
    coord[:, :geom_opt_idx + 1]), delimiter=',')
np.savetxt('bond_hist.csv', abs(
    coord[0, :geom_opt_idx + 1] - coord[1, :geom_opt_idx + 1]), delimiter=',')