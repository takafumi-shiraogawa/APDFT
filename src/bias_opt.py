#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
import numpy as np
import gc

# Algorithms
# 1. Identify unique molecules


def get_target_value(target, dict_inp, apdft_order):
  target_values = []
  for i, row in enumerate(dict_inp):
    target_values.append(row["%s%s" % (target, str(apdft_order))])

  return target_values


# Obtain nuclear-nuclear repulsion energies of all molecules in chemcal space
inp_nuc_energies = open("./nuc_energies.csv", "r")
dict_inp_nuc_energies = csv.DictReader(inp_nuc_energies)
full_nuc_energies = get_target_value("nuc_energy_order", dict_inp_nuc_energies, 0)
inp_nuc_energies.close()

# Obtain indexes of the unique molecules
unique_nuc_energies, id_unique_mol = np.unique(
    full_nuc_energies, return_index=True, axis=0)

del full_nuc_energies
del unique_nuc_energies
gc.collect()
