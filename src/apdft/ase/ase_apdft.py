import os
import shutil
import jinja2 as jinja
import csv
import numpy as np
from ase.calculators.calculator import FileIOCalculator

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

  # Set n2.xyz or n2_mod.inp with arbitrary bond length
  # TODO: generalization to three Cartesian coordinates
  def set_inputs(target1, target2, target_inp):
    with open(str(target_inp)) as fh:
        template = jinja.Template(fh.read())

    env = {}
    env["bond_length1"] = target1
    env["bond_length2"] = target2

    return template.render(**env)

  def gener_inputs(coord):
    # TODO: generalization to three Cartesian coordinates
    inputfile_ori = handle_APDFT.set_inputs(coord[0], coord[1], "template/n2.xyz")
    inputfile_mod = handle_APDFT.set_inputs(
        coord[0], coord[1], "template/n2_mod.xyz")

    with open("work/temp/n2.xyz", "w") as inp:
      inp.write(inputfile_ori)
    with open("work/temp/n2_mod.xyz", "w") as inp:
      inp.write(inputfile_mod)

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

    # This is a bond length
    coord = np.zeros(len(self.atoms.positions))

    # TODO: generalization to three Cartesian coordinates
    for i in range(len(self.atoms.positions)):
      coord[i] = self.atoms.positions[i, 2]

    handle_APDFT.gener_inputs(coord)

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
    # TODO: generalization to three Cartesian coordinates
    for i in range(num_atoms):
      atom_forces[i, 2] = handle_APDFT.get_target_value(
          "ver_atomic_force_%s_order" % str(i), dict_atomic_force, apdft_order)
      inp_atomic_force.close()
      inp_atomic_force = open("%s/ver_atomic_forces.csv" % path, "r")
      dict_atomic_force = csv.DictReader(inp_atomic_force)

    inp_total_energy.close()
    inp_atomic_force.close()

    # TODO: generalization to three Cartesian coordinates
    print("APDFT results:", self.num_opt_step, pot_energy, self.atoms.positions[1, 2] - self.atoms.positions[0, 2])

    self.results = {'energy': pot_energy * har_to_ev,
                    'forces': atom_forces * hb_to_ea,
                    'stress': np.zeros(6),
                    'dipole': np.zeros(3),
                    'charges': np.zeros(num_atoms),
                    'magmom': 0.0,
                    'magmoms': np.zeros(num_atoms)}
    handle_APDFT.save_results(self.num_opt_step)