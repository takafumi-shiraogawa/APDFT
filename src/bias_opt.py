#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
import numpy as np
import gc

# Algorithms
# 1. Identify unique molecules
# 2. Generate weights of the molecules


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

def calc_weight_energy_and_gradients(num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights):
  # Set information on outputs of the APDFT calculation
  inp_total_energy = open("./energies.csv", "r")
  inp_atomic_force = open("./ver_atomic_forces.csv", "r")

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
    inp_atomic_force = open("./ver_atomic_forces.csv", "r")
    dict_atomic_force = csv.DictReader(inp_atomic_force)

  full_gradients[:, :] = -full_gradients[:, :]

  # Compute weighted properties
  weight_energy = get_weight_property(full_energies, id_unique_mol, mol_weights)
  for i in range(num_atom):
    weight_gradients[i] = get_weight_property(
        full_gradients[:, i], id_unique_mol, mol_weights)

  return weight_energy, weight_gradients

def calc_weight_dipole(num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights):
  # Set information on outputs of the APDFT calculation
  inp_dipole = open("./dipoles.csv", "r")

  # Open the inputs
  dict_dipole = csv.DictReader(inp_dipole)

  full_dipoles = np.zeros(num_full_mol)

  # Obtain results
  full_dipoles = get_target_value(
      "dipole_moment_z_order", dict_dipole, apdft_order)

  # Compute weighted properties
  weight_dipole = get_weight_property(
      full_dipoles, id_unique_mol, mol_weights)

  return weight_dipole


# Parameters
apdft_order = 1
num_atom = 2


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

weight_energy, weight_gradients = calc_weight_energy_and_gradients(
    num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights)

weight_dipole = calc_weight_dipole(
    num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights)
print(weight_dipole)