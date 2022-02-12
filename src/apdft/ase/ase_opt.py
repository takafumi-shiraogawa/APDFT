from ase import Atoms
# from ase.optimize import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
import apdft as APDFTtool
import apdft.ase.ase_apdft as APDFT
import time

# Input
# init.xyz : initial coordinates of optimization
# template/ : it is used in APDFT interface (APDFT here) and contains
#             apdft.conf imp_mod_cli1.sh imp_mod_cli2.sh n2.xyz n2_mod.xyz

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


  def imp_ase_opt():

    start = time.time()

    # coordinates of init.xyz should be in Angstrom.
    nuclear_numbers, coordinates = APDFTtool.read_xyz("init.xyz")

    molstring = ASE_OPT.get_molstring(nuclear_numbers)

    MOL = Atoms(molstring,
              positions=coordinates,
              calculator=APDFT.mod_APDFT())

    # dyn = BFGS(MOL)
    dyn = BFGSLineSearch(MOL)
    dyn.run(fmax=0.005 * hb_to_ea)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    with open('elapsed_time.dat', 'w') as tfile:
      tfile.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")