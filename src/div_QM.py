#!/usr/bin/env python

# This code was taken from bias_opt.py

import os
import sys
from multiprocessing import Process

# How to use?
# python3 div_QM.py 3
# Here 3 is parallerization variable


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


par_var = sys.argv[1]
par_var = int(par_var)
path = "."

real_par_var = gener_commands_file(path, par_var)
real_par_var = int(real_par_var)
processes = [
      Process(target=inp_commands_file, args=(path, i))
      for i in range(real_par_var)]
for p in processes:
    p.start()
for p in processes:
    p.join()
