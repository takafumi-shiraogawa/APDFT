import os
from ase import Atoms
from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
import apdft as APDFTtool
import apdft.ase.ase_apdft as APDFT
from apdft.ase.steepest_descent import STEEPEST_DESCENT
import time

# Input
# init.xyz : initial coordinates of optimization
# template/ : it is used in APDFT interface (APDFT here) and contains
#             apdft.conf imp_mod_cli1.sh imp_mod_cli2.sh

# How to use?
# $ python3 test.py
# test.py:
#   import apdft.ase.ase_opt as ase_opt
#   ase_opt.ASE_OPT.imp_ase_opt()

class ASE_OPT():
  """ implement ASE optimizer."""

  # Conversion factor from Angstrom to Bohr
  ang_to_bohr = 1 / 0.52917721067
  # Conversion factor from hartree to eV
  har_to_ev = 27.21162

  # Hartree / Bohr to eV / Angstrom
  hb_to_ea = har_to_ev * ang_to_bohr

  def get_molstring(nuclear_numbers):
    nuclear_symbols = []

    # Only can deal with H, He, B, C, N, O, F
    for i in range(len(nuclear_numbers)):
      if nuclear_numbers[i] == 1:
        nuclear_symbols.append("H")
      elif nuclear_numbers[i] == 2:
        nuclear_symbols.append("He")
      elif nuclear_numbers[i] == 5:
        nuclear_symbols.append("B")
      elif nuclear_numbers[i] == 6:
        nuclear_symbols.append("C")
      elif nuclear_numbers[i] == 7:
        nuclear_symbols.append("N")
      elif nuclear_numbers[i] == 8:
        nuclear_symbols.append("O")
      elif nuclear_numbers[i] == 9:
        nuclear_symbols.append("F")

    return ''.join(nuclear_symbols)


  def imp_ase_opt(fmax_au=0.005, optimizer=None):

    start = time.time()

    # coordinates of init.xyz should be in Angstrom.
    nuclear_numbers, coordinates = APDFTtool.read_xyz("init.xyz")

    molstring = ASE_OPT.get_molstring(nuclear_numbers)

    MOL = Atoms(molstring,
              positions=coordinates,
              calculator=APDFT.mod_APDFT())

    # Remove an old results of geometry optimization
    if os.path.isfile('BFGSLineSearch.dat'):
      os.remove('BFGSLineSearch.dat')

    if optimizer is None:
      dyn = BFGSLineSearch(MOL, logfile="BFGSLineSearch.dat", maxstep=0.1)
    elif optimizer == "BFGS":
      dyn = BFGS(MOL, logfile="BFGSLineSearch.dat", maxstep=0.1)
    elif optimizer == "STEEPEST_DESCENT":
      dyn = STEEPEST_DESCENT(MOL, logfile="STEEPEST_DESCENT.dat", maxstep=0.1)
    else:
      raise ValueError("Specification of the optimizer is invalid.")
    dyn.run(fmax=fmax_au * ASE_OPT.hb_to_ea)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    with open('elapsed_time.dat', 'w') as tfile:
      tfile.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")