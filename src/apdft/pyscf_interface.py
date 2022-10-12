import os
import numpy as np
import pyscf.gto
import pyscf.tools.cubegen

def _format_coordinates(nuclear_numbers, coordinates):
    """ Converts the vector representation into the atomspec format for PySCF."""
    ret = []
    for z, pos in zip(nuclear_numbers, coordinates):
        ret.append("%d %.15f %.15f %.15f" % (z, *pos))
    return ";".join(ret)

class PySCF_Mol():
  """ Processigng PySCF in APDFT. """
  # Ad hoc implementation, it needs to be modified for a large number of
  # calculations.

  def __init__(self, nuclear_number, nuclear_coordinate, div_elements):
    self._nuclear_number = nuclear_number
    self._nuclear_coordinate = nuclear_coordinate

    mol = pyscf.gto.Mole()
    mol.atom = _format_coordinates(
        self._nuclear_number, self._nuclear_coordinate)
    mol.build()

    self._mol = mol
    self._div_elements = div_elements

  def read_cube(self, pos_cube):
    """ Read a cube file.
    Args:
      pos_cube : A string of the cube directory.
    Returns:
      A numpy array of the coordinates of the grids of the density in the dimension (div_elements^3, 3).
      A numpy array of the real-space density amplitudes in the dimension (div_elements, div_elements, div_elements).
    """
    # For N2
    margin = 5.0
    # For benzene
    # margin = 6.0
    cube = pyscf.tools.cubegen.Cube(
        self._mol, nx=self._div_elements, ny=self._div_elements, nz=self._div_elements, resolution=None, margin=margin, origin=None)

    file_cube = "%s%s" % (str(pos_cube), "/cubegen.cube")

    # Note that after cube.read(file_cube), cube.get_coords() returns a strange
    # array.
    return cube.get_coords(), cube.read(file_cube)

  def write_cube(self, cube_density, cube_dir, cube_file_name):
    """
    Args:
      cube_density : A numpy array of float.
      cube_dir : A string of a cube file directory.
      cube_file_name : A string of an output cube file.
    """
    # For N2
    margin = 5.0
    # For benzene
    # margin = 6.0
    cube = pyscf.tools.cubegen.Cube(
        self._mol, nx=self._div_elements, ny=self._div_elements, nz=self._div_elements, resolution=None, margin=margin, origin=None)

    cube_file_name = "%s%s%s" % (cube_dir, cube_file_name, ".cube")

    cube.write(cube_density, cube_file_name, comment="Electron density")

    return