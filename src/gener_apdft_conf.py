#!/usr/bin/env python

# A code for generating apdft.conf with different fractional numbers
# for finite differential in the perturbed electron density

# Requirements:
# In the curent directory,
# ./order.inp
# ./template/
#   apdft.conf.template
#   commands.sh
#   imp_mod_cli1.sh
#   imp_mod_cli2.sh
#   n2.xyz
#   n2_mod.xyz

import os
import shutil
import jinja2 as jinja


# Get apdft.conf with variables of fractional numbers for
# finite differential in the perturbed electron density
def gener_inputs(deltaZ, deltaR):
  # Obtain path of the current directory.
  # basedir = os.path.dirname(os.path.abspath(__file__))
  # with open("%s/template/apdft.conf.template" % basedir) as fh:
  with open("template/apdft.conf.template") as fh:
    template = jinja.Template(fh.read())

  env = {}
  env["deltaZ"] = deltaZ
  env["deltaR"] = deltaR

  return template.render(**env)


def copy_ingredients(order1, order2):
  # basedir = os.path.dirname(os.path.abspath(__file__))
  # copyfile = "%s/template/commands.sh" % basedir

  # Set a target directory
  copy_directory = "QM/delta-%s-%s" % (str(order1), str(order2))

  # Copy commands.sh
  copyfile = "template/commands.sh"
  shutil.copy(copyfile, copy_directory)

  # Copy imp_mod_cli1.sh
  copyfile = "template/imp_mod_cli1.sh"
  shutil.copy(copyfile, copy_directory)

  # Copy imp_mod_cli2.sh
  copyfile = "template/imp_mod_cli2.sh"
  shutil.copy(copyfile, copy_directory)

  # Copy n2.xyz
  copyfile = "template/n2.xyz"
  shutil.copy(copyfile, copy_directory)

  # Copy n2_mod.xyz
  copyfile = "template/n2_mod.xyz"
  shutil.copy(copyfile, copy_directory)

  return


# If QM/ exists, it is removed.
if os.path.isdir("QM/"):
  shutil.rmtree("QM/")

# If all_commands.sh exists, it is removed.
if os.path.isfile("all_commands.sh"):
  os.remove("all_commands.sh")

# If all_imp_mod_cli1.sh exists, it is removed.
if os.path.isfile("all_imp_mod_cli1.sh"):
  os.remove("all_imp_mod_cli1.sh")

# If all_imp_mod_cli2.sh exists, it is removed.
if os.path.isfile("all_imp_mod_cli2.sh"):
  os.remove("all_imp_mod_cli2.sh")

# Read specified variables dR or dZ
order_inp = open('order.inp', 'r')
specified_var = order_inp.read()
if specified_var != "Z" and specified_var != "R":
  raise ValueError("Error! order.inp should be Z or R!")

# Set parameters.
change_frac_num = 5.0 * (10.0 ** (-8))
eval_num = 7
# That is, the number of evaluations is 140, and
# the range is from 10 ** (-7) to 10 ** 0.

count = 0

# For all_commands.sh, all_imp_mod_cli1.sh, and all_imp_mod_cli2.sh
all_commands = []
all_imp1 = []
all_imp2 = []

print('')
print(specified_var)

for i in range(eval_num):
  effect_change_frac_num = change_frac_num * (10.0 ** i)

  for j in range(20):
    count += 1
    delta = effect_change_frac_num * (j + 1)

    # For check
    print(count, i, j, delta)

    # Make directries
    path = "QM/delta-%s-%s" % (i + 1, j + 1)
    os.makedirs(path)

    # Make an input file
    if specified_var == "Z":
      inputfile = gener_inputs(delta, 0.005)
    elif specified_var == "R":
      # 0.1 is a factor for preventing overlap of atoms.
      inputfile = gener_inputs(0.05, delta * 0.1)

    with open("%s/apdft.conf" % path, "w") as inp:
      inp.write(inputfile)

    # Copy commands.sh
    copy_ingredients(i + 1, j + 1)

    # For all_commands.sh, all_imp_mod_cli1.sh, and all_imp_mod_cli2.sh
    all_commands.append("( cd %s && bash commands.sh )" % path)
    all_imp1.append("( cd %s && bash imp_mod_cli1.sh )" % path)
    all_imp2.append("( cd %s && bash imp_mod_cli2.sh )" % path)


print('')

# Write all_commands.sh
with open("all_commands.sh", "w") as fh:
  fh.write("\n".join(all_commands))

# Write all_imp_mod_cli1.sh
with open("all_imp_mod_cli1.sh", "w") as fh:
  fh.write("\n".join(all_imp1))

# Write all_imp_mod_cli2.sh
with open("all_imp_mod_cli2.sh", "w") as fh:
  fh.write("\n".join(all_imp2))
