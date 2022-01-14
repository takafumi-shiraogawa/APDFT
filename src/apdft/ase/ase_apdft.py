import os
import shutil
import jinja2 as jinja
import csv
import numpy as np
from ase.calculators.calculator import FileIOCalculator
import apdft

# Conversion factor from Angstrom to Bohr
ang_to_bohr = 1 / 0.52917721067
# Conversion factor from hartree to eV
har_to_ev = 27.21162

# Hartree / Bohr to eV / Angstrom
hb_to_ea = har_to_ev * ang_to_bohr

class handle_APDFT():

  # Copy inputs of an APDFT calculation from template/ to the target directry
  def copy_ingredients():

    # Set a target directry
    copy_directry = "work/temp"

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

  # Read initial geometry of a molecule
  # The name of the xyz file is init.xyz
  def read_init_geom():
    try:
        nuclear_numbers, coordinates = apdft.read_xyz("init.xyz")
        return nuclear_numbers, coordinates
    except FileNotFoundError:
        apdft.log.log(
            'Unable to open input file "%s".' % "init.xyz", level="error"
        )
        return

  # Set mol.xyz or mol_mod.inp from the uptdated coordinates
  def set_inputs(nuclear_numbers, coordinates, target_inp):
    with open(str(target_inp), mode='w') as fh:
      print(len(nuclear_numbers), file=fh)
      print("", file=fh)
      for i in range(len(nuclear_numbers)):
        print(nuclear_numbers[i], *coordinates[i], file=fh)

  # generate inputs of APDFT calculations
  def gener_inputs(nuclear_numbers, coordinates):
    handle_APDFT.set_inputs(nuclear_numbers, coordinates, "work/temp/n2.xyz")
    handle_APDFT.set_inputs(nuclear_numbers, coordinates, "work/temp/n2_mod.xyz")

  # Read target energy from the output (energies.csv)
  def get_target_value(target, dict_inp, apdft_order):

    for i, row in enumerate(dict_inp):
      # Read 1st line of the output of total energy
      if i == 0:
        return np.array(row["%s%s" % (str(target), str(apdft_order))], dtype=np.float64)

      else:
        raise ValueError("Error in reading target values!")

  # Make a work directry
  def make_work():
    path = "work/temp"
    os.makedirs(path)

  # Make a work directry to save results
  def make_work_save(num_opt_step):
    path = "work/iter-%s" % num_opt_step
    os.makedirs(path)

  # Make a work directry to save results
  def save_results(num_opt_step):
    # handle_APDFT.make_work_save(num_opt_step)

    path = "work/iter-%s" % num_opt_step

    shutil.copytree('./work/temp', path)


# Add modified APDFT to ASE calculators
# Refer to https://wiki.fysik.dtu.dk/ase/development/calculators.html
class mod_APDFT(FileIOCalculator):
  implemented_properties = ['energy', 'forces']

  # command = "( cd work/temp && bash imp_mod_cli1.sh && bash commands.sh && bash imp_mod_cli2.sh )"
  command = "( cd work/temp && bash imp_mod_cli1.sh && div_QM.py 8 && bash imp_mod_cli2.sh)"
  discard_results_on_any_change = True

  def __init__(self, *args, label='APDFT', num_opt_step = None, **kwargs):
    FileIOCalculator.__init__(self, *args, label=label, **kwargs)
    self.num_opt_step = num_opt_step

    # Get nuclear numbers
    # Note that APDFT can work with nuclear numbers
    # For nuclear coordinates, self.atoms.positions can be used instead.
    self.nuclear_numbers, coordinates = handle_APDFT.read_init_geom()

    if os.path.isdir("work/"):
      shutil.rmtree("work/")


  def write_input(self, atoms, properties=None, system_changes=None):
    if os.path.isdir("work/temp/"):
      shutil.rmtree("work/temp/")

    if self.num_opt_step is None:
      self.num_opt_step = 1
    else:
      self.num_opt_step += 1

    # Make a working directory
    handle_APDFT.make_work()

    handle_APDFT.copy_ingredients()

    # handle_APDFT.gener_inputs(coord)
    handle_APDFT.gener_inputs(self.nuclear_numbers, self.atoms.positions)

  # Read calculated energy and atomic forces
  def read_results(self):
    path = 'work/temp'

    # Set information on outputs of the APDFT calculation
    inp_total_energy = open("%s/energies.csv" % path, "r")
    inp_atomic_force = open("%s/ver_atomic_forces.csv" % path, "r")

    # Open the inputs
    dict_total_energy = csv.DictReader(inp_total_energy)
    dict_atomic_force = csv.DictReader(inp_atomic_force)

    num_atoms = len(self.atoms.positions)

    pot_energy = 0
    atom_forces = np.zeros((num_atoms, 3))

    apdft_order = 1

    # Obtain results
    pot_energy = handle_APDFT.get_target_value(
        "total_energy_order", dict_total_energy, apdft_order)

    # For full-dimensional Cartesian optimization
    for i in range(num_atoms):
      for didx, dim in enumerate("xyz"):
        try:
          atom_forces[i, didx] = handle_APDFT.get_target_value(
              "ver_atomic_force_%s_%s_order" % (str(i), dim), dict_atomic_force, apdft_order)
        except FileNotFoundError:
          print(FileNotFoundError)
        except KeyError:
          # For z-Cartesian component
          if didx == 2:
            inp_atomic_force.close()
            inp_atomic_force = open("%s/ver_atomic_forces.csv" % path, "r")
            dict_atomic_force = csv.DictReader(inp_atomic_force)
            atom_forces[i, 2] = handle_APDFT.get_target_value(
              "ver_atomic_force_%s_order" % str(i), dict_atomic_force, apdft_order)
          # For x- and y-Cartesian components
          else:
            atom_forces[i, didx] = 0.0
        inp_atomic_force.close()
        inp_atomic_force = open("%s/ver_atomic_forces.csv" % path, "r")
        dict_atomic_force = csv.DictReader(inp_atomic_force)

    inp_total_energy.close()
    inp_atomic_force.close()

    print("APDFT results:", self.num_opt_step, pot_energy)

    self.results = {'energy': pot_energy * har_to_ev,
                    'forces': atom_forces * hb_to_ea,
                    'stress': np.zeros(6),
                    'dipole': np.zeros(3),
                    'charges': np.zeros(num_atoms),
                    'magmom': 0.0,
                    'magmoms': np.zeros(num_atoms)}
    handle_APDFT.save_results(self.num_opt_step)