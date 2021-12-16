#!/usr/bin/env python

import csv
import numpy as np
import gc
import os
import shutil
import jinja2 as jinja
# from basis_set_exchange import lut
from multiprocessing import Process

# Inputs
# order.inp
# nuc_energies.xyz
# mol.xyz
# mol.xyz needs to have numbers for atom species.

# Algorithms
# 1. Identify unique molecules
# 2. Generate weights of the molecules
# 3. Geometry optimization
# 4. Back to 2

# def get_element_number(element):
#     return lut.element_Z_from_sym(element)

def read_xyz(fn):
    with open(fn) as fh:
        lines = fh.readlines()
    numatoms = int(lines[0].strip())
    lines = lines[2 : 2 + numatoms]
    nuclear_numbers = []
    coordinates = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            break
        parts = line.split()
        nuclear_numbers.append(int(parts[0]))
        # try:
        #     nuclear_numbers.append(int(parts[0]))
        # except:
        #     nuclear_numbers.append(get_element_number(parts[0]))
        coordinates.append([float(_) for _ in parts[1:4]])
    return np.array(nuclear_numbers), np.array(coordinates)

# Copy inputs of an APDFT calculation from template/ to the target directry
def copy_ingredients(copy_directry):

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
def gener_inputs(targets, target_inp):
  with open(str(target_inp)) as fh:
    template = jinja.Template(fh.read())

  env = {}
  for target_idx, target in enumerate(targets):
    env["bond_length%s" % str(target_idx + 1)] = target

  return template.render(**env)

def get_target_value(target, dict_inp, apdft_order):
  target_values = []
  for i, row in enumerate(dict_inp):
    target_values.append(row["%s%s" % (target, str(apdft_order))])

  return np.array(target_values).astype(np.float64)

def get_target_value_wo_order(target, dict_inp):
  target_values = []
  for i, row in enumerate(dict_inp):
    target_values.append(row["%s" % target])

  return np.array(target_values).astype(str)

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

def get_unique_mols(full_list_mols, id_unique_mol):
  unique_list_mols = []
  pos = -1
  for i in range(len(full_list_mols)):
    if i in id_unique_mol:
      pos += 1
      unique_list_mols.append(full_list_mols[i])

  return unique_list_mols

def calc_weights_from_dipoles(local, sigma, path, num_full_mol, num_atom, apdft_order, id_unique_mol):
  # Set information on outputs of the APDFT calculation
  inp_dipole = open("%s/dipoles.csv" % path, "r")

  # Open the inputs
  dict_dipole = csv.DictReader(inp_dipole)

  full_dipoles = np.zeros(num_full_mol)

  # Obtain results
  full_dipoles = get_target_value(
      "dipole_moment_z_order", dict_dipole, apdft_order)

  prop_sum = 0.0
  unique_dipoles = np.zeros(len(id_unique_mol))
  pos = -1
  for i in range(num_full_mol):
    if i in id_unique_mol:
      pos += 1
      unique_dipoles[pos] = full_dipoles[i]

  weights = np.zeros(len(id_unique_mol))
  if not local:
    prop_sum = np.exp(sigma * (abs(unique_dipoles) ** 2.0)).sum()
    for i in range(len(id_unique_mol)):
      weights[i] = np.exp(sigma * (abs(unique_dipoles[i]) ** 2.0)) / prop_sum
  else:
    weights[:] = 0.0
    weights[np.argmax(abs(unique_dipoles))] = 1.0

  inp_dipole.close()

  return weights

# Conduct line search
def line_search(coord, energy, gradient):
  next_coord = np.zeros(len(coord))
  for i in range(len(coord)):
    # next_coord[i] = coord[i] - (energy / gradient[i])
    next_coord[i] = coord[i] - (0.2 * gradient[i])

  return next_coord

def readlines_commands_file(path):
    with open('%s/commands.sh' % path, 'r') as file:
        return file.readlines()

def save_commands_file(file_name, text):
    with open(file_name, 'w') as file:
      file.write(text)

def gener_commands_file(path):
  # Read parallerization variable
  order_inp = open('order.inp', 'r')
  par_var = order_inp.read()
  par_var = int(par_var)

  commands_lines = readlines_commands_file(path)

  div_num = len(commands_lines) // par_var
  if len(commands_lines) % par_var == 0:
    tune_div_num = div_num
  else:
    tune_div_num = div_num + 1
  div_commands_lines = [commands_lines[i:i+tune_div_num]
                        for i in range(0, len(commands_lines), tune_div_num)]

  for textidx, text in enumerate(div_commands_lines):
    save_commands_file("%s/commands_%s.sh" % (path, str(textidx)), "".join(text))

  if len(commands_lines) % tune_div_num == 0:
    return len(commands_lines) // tune_div_num
  else:
    return (len(commands_lines) // tune_div_num) + 1

def inp_commands_file(path, pos):
  os.system("( cd %s && bash commands_%s.sh )" % (path, str(pos)))


# Begin the code
if os.path.isdir("work/"):
  shutil.rmtree("work/")

if os.path.isfile('log'):
  os.remove('log')

# For log
log = open('log', 'w')

nuclear_numbers, coordinates = read_xyz("mol.xyz")

# Parameters
apdft_order = 1
num_atom = len(nuclear_numbers)
# maximum number of optimization loops
max_geom_opt = 30
ipsilon = 0.01
sigma = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0]).astype(np.float64)

# Convertor
ang_to_bohr = 1.0 / 0.529117

# Results
# APDFT energy
energy = np.zeros((len(sigma) + 1, max_geom_opt))
# APDFT gradient
gradient = np.zeros((num_atom, len(sigma) + 1, max_geom_opt))
# Atomic coordinates
coord = np.zeros((num_atom, len(sigma) + 1, max_geom_opt))
dipole = np.zeros((len(sigma) + 1, max_geom_opt))

for i in range(num_atom):
  coord[i, 0, 0] = coordinates[i, 2]

# Obtain nuclear-nuclear repulsion energies of all molecules in chemcal space
inp_nuc_energies = open("./nuc_energies.csv", "r")
dict_inp_nuc_energies = csv.DictReader(inp_nuc_energies)
full_nuc_energies = get_target_value("nuc_energy_order", dict_inp_nuc_energies, 0)
inp_nuc_energies.close()
inp_nuc_energies = open("./nuc_energies.csv", "r")
dict_inp_nuc_energies = csv.DictReader(inp_nuc_energies)
full_list_mol = get_target_value_wo_order("targets", dict_inp_nuc_energies)
# Consider numerical errors
full_nuc_energies = np.round(full_nuc_energies, decimals=10)
inp_nuc_energies.close()

num_full_mol = len(full_nuc_energies)

# Obtain indexes of the unique molecules
unique_nuc_energies, id_unique_mol = np.unique(
    full_nuc_energies, return_index=True, axis=0)

unique_list_mol = get_unique_mols(full_list_mol, id_unique_mol)
unique_list_mol = np.array(unique_list_mol, dtype=str)
# Save results
np.savetxt('unique_list_mol.csv', zip(id_unique_mol, unique_list_mol), delimiter=',', fmt="%s")

del full_nuc_energies
del unique_nuc_energies
gc.collect()

# The number of unique molecules
num_unique_mol = len(id_unique_mol)

# Set initial weights
mol_weights = np.zeros(len(id_unique_mol))

save_geom_opt_idx = np.zeros(len(sigma) + 1).astype(np.int64)

for bias_shift_idx in range(len(sigma) + 1):

  if bias_shift_idx == 0:
    mol_weights[:] = 1.0 / num_unique_mol
  elif bias_shift_idx < len(sigma):
    local = False
    mol_weights = calc_weights_from_dipoles(
        local, sigma[bias_shift_idx], former_path, num_full_mol, num_atom, apdft_order, id_unique_mol)
  else:
    local = True
    # 1.0 is not used here
    mol_weights = calc_weights_from_dipoles(
        local, 1.0, former_path, num_full_mol, num_atom, apdft_order, id_unique_mol)

    # Save results
    np.savetxt('unique_list_design_mol.csv', zip(id_unique_mol, unique_list_mol, mol_weights), delimiter=',', fmt="%s")

  for geom_opt_idx in range(max_geom_opt):

    if bias_shift_idx > 0 and geom_opt_idx == 0:
      coord[:, bias_shift_idx, geom_opt_idx] = coord[:,
                                                     bias_shift_idx - 1, save_geom_opt_idx[bias_shift_idx - 1]]

    path = "work/bias-iter-%s/opt-iter-%s" % (
        str(bias_shift_idx), str(geom_opt_idx))
    os.makedirs(path)

    # Copy inputs of the APDFT calculation in the working directry
    copy_ingredients(path)

    # Set *.xyz
    inputfile_ori = gener_inputs(
        coord[:, bias_shift_idx, geom_opt_idx], "template/n2.xyz")
    inputfile_mod = gener_inputs(
        coord[:, bias_shift_idx, geom_opt_idx], "template/n2_mod.xyz")
    with open("%s/n2.xyz" % path, "w") as inp:
      inp.write(inputfile_ori)
    with open("%s/n2_mod.xyz" % path, "w") as inp:
      inp.write(inputfile_mod)

    # Implement the APDFT calculation
    os.system("( cd %s && bash imp_mod_cli1.sh )" % path)

    # os.system("( cd %s && bash commands.sh )" % path)
    real_par_var = gener_commands_file(path)
    real_par_var = int(real_par_var)
    if __name__ == "__main__":
      processes = [
          Process(target=inp_commands_file, args=(path, i))
          for i in range(real_par_var)]
      for p in processes:
          p.start()
      for p in processes:
          p.join()

    os.system("( cd %s && bash imp_mod_cli2.sh )" % path)

    weight_energy, weight_gradients = calc_weight_energy_and_gradients(
        path, num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights)

    weight_dipole = calc_weight_dipole(
        path, num_full_mol, num_atom, apdft_order, id_unique_mol, mol_weights)

    energy[bias_shift_idx, geom_opt_idx] = weight_energy
    gradient[:, bias_shift_idx, geom_opt_idx] = weight_gradients
    dipole[bias_shift_idx, geom_opt_idx] = weight_dipole

    print("")
    print("*** RESULTS ***")
    print("Bias step", bias_shift_idx)
    print("Opt step", geom_opt_idx)
    print("Energy", energy[bias_shift_idx, geom_opt_idx])
    if geom_opt_idx > 0:
      print("Energy difference",
            energy[bias_shift_idx, geom_opt_idx] - energy[bias_shift_idx, geom_opt_idx - 1])
    print("Gradient", gradient[:, bias_shift_idx, geom_opt_idx])
    print("Coordinates", coord[:, bias_shift_idx, geom_opt_idx])
    # print("Bond length", abs(
    #     coord[0, bias_shift_idx, geom_opt_idx] - coord[1, bias_shift_idx, geom_opt_idx]))
    print("Dipole", dipole[bias_shift_idx, geom_opt_idx])
    print("")

    if bias_shift_idx > 0 and geom_opt_idx == 0:
      log.write("\n")
      log.write("\n")
      log.write("\n")
      log.write("\n")

    log.write("Step-%s-%s (bias-opt)\n" %
              (str(bias_shift_idx), str(geom_opt_idx)))
    log.write("Energy, %s\n" % str(energy[bias_shift_idx, geom_opt_idx]))
    if geom_opt_idx > 0:
      log.write("Energy difference, %s\n" % str(
          energy[bias_shift_idx, geom_opt_idx] - energy[bias_shift_idx, geom_opt_idx - 1]))
    log.write("Gradient, %s\n" %
              str(gradient[:, bias_shift_idx, geom_opt_idx]))
    log.write("Coordinates, %s\n" %
              str(coord[:, bias_shift_idx, geom_opt_idx]))
    # log.write("Bond length, %s\n" % str(abs(
    #     coord[0, bias_shift_idx, geom_opt_idx] - coord[1, bias_shift_idx, geom_opt_idx])))
    log.write("Dipole, %s\n" % str(abs(dipole[bias_shift_idx, geom_opt_idx])))
    log.write("\n")


    # Save results
    np.savetxt('energy_hist.csv', energy[bias_shift_idx, :geom_opt_idx + 1])
    np.savetxt('grad_hist.csv', np.transpose(
        gradient[:, bias_shift_idx, :geom_opt_idx + 1]), delimiter=',')
    np.savetxt('geom_hist.csv', np.transpose(
        coord[:, bias_shift_idx, :geom_opt_idx + 1]), delimiter=',')
    # np.savetxt('bond_hist.csv', abs(
    #     coord[0, bias_shift_idx, :geom_opt_idx + 1] - coord[1, bias_shift_idx, :geom_opt_idx + 1]), delimiter=',')

    if np.amax(abs(gradient[:, bias_shift_idx, geom_opt_idx])) < ipsilon:
      former_path = path
      save_geom_opt_idx[bias_shift_idx] = geom_opt_idx

      # Save results
      np.savetxt('energy_hist_bias%s.csv' % str(bias_shift_idx),
                 energy[bias_shift_idx, :geom_opt_idx + 1])
      np.savetxt('grad_hist_bias%s.csv' % str(bias_shift_idx), np.transpose(
          gradient[:, bias_shift_idx, :max(save_geom_opt_idx) + 1]), delimiter=',')
      np.savetxt('geom_hist_bias%s.csv' % str(bias_shift_idx), np.transpose(
          coord[:, bias_shift_idx, :max(save_geom_opt_idx) + 1]), delimiter=',')
      # np.savetxt('bond_hist_bias%s.csv' % str(bias_shift_idx), abs(
      #     coord[0, bias_shift_idx, :max(save_geom_opt_idx) + 1] - coord[1, bias_shift_idx, :max(save_geom_opt_idx) + 1]), delimiter=',')
      np.savetxt('dipole_hist_bias%s.csv' % str(bias_shift_idx),
          dipole[bias_shift_idx, :max(save_geom_opt_idx) + 1], delimiter=',')

      break

    # Perform line search and obtain renewed coordinates
    coord[:, bias_shift_idx, geom_opt_idx + 1] = line_search(coord[:, bias_shift_idx, geom_opt_idx],
                                                                 energy[bias_shift_idx, geom_opt_idx], gradient[:, bias_shift_idx, geom_opt_idx] / ang_to_bohr)

log.close()
