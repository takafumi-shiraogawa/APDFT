#!/usr/bin/env python

# This code was taken from bias_opt.py

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import os

# How to use?
# python3 div_QM.py 3
# Here 3 is parallerization variable

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

def readlines_commands_file(path):
    with open('%s/commands.sh' % path, 'r') as file:
        return file.readlines()

def save_commands_file(file_name, text):
    with open(file_name, 'w') as file:
      file.write(text)

def gener_commands_file(path, par_var):
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


if mpi_rank == 0:

  par_var = mpi_size - 20
  path = "."

  real_par_var = gener_commands_file(path, par_var)
  real_par_var = int(real_par_var)

if __name__ == "__main__":
  with MPIPoolExecutor(max_workers=real_par_var) as executor:
    process_args = [(path, i) for i in range(real_par_var)]
    executor.map(inp_commands_file, process_args)