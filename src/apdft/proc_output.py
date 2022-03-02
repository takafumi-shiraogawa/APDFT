class Geom_Output():
  """ Generate geometry outputs of molecules. """

  def xyz_output(nuclear_number, coord, name):
    """ Generate *.xyz. """
    natom = len(nuclear_number)

    if natom != len(coord):
      raise ValueError("Error: generation of *.xyz")

    with open("%s%s" % (str(name), ".xyz"), mode='w') as fh:
      # Write headers
      print(natom, file=fh)

      # Write a last brank line
      print("", file=fh)

      # Write nuclear coordinates
      for i in range(natom):
        print(nuclear_number[i], *coord[i, :], file=fh)

      # Write a last brank line
      print("", file=fh)

  def gjf_output(nuclear_number, coord, name):
    """ Generate *.gjf. """
    natom = len(nuclear_number)

    if natom != len(coord):
      raise ValueError("Error: generation of *.gjf")

    with open("%s%s" % (str(name), ".gjf"), mode='w') as fh:
      # Write headers
      print("%mem = 10GB", file=fh)
      print("%nproc = 6", file=fh)
      print("# hf/def2tzvp opt freq nosymm", file=fh)
      print("", file=fh)
      print(*nuclear_number[:], file=fh)
      print("", file=fh)
      # Here assuming neutral and spin-singlet molecule
      print("0 1", file=fh)

      # Write nuclear coordinates
      for i in range(natom):
        print(nuclear_number[i], *coord[i, :], file=fh)

      # Write a last brank line
      print("", file=fh)