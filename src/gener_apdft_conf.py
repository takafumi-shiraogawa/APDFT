#!/usr/bin/env python

# A code for generating apdft.conf with different fractional numbers
# for finite differential in the perturbed electron density

# Requirements:
# In the curent directry,
# template/
#   apdft.conf.template
#   commands.sh

import os
import shutil
import jinja2 as jinja


# Get apdft.conf with variables of fractional numbers for
# finite differential in the perturbed electron density
def gener_inputs(deltaZ, deltaR):
  # Obtain path of the current directry.
  # basedir = os.path.dirname(os.path.abspath(__file__))
  # with open("%s/template/apdft.conf.template" % basedir) as fh:
  with open("template/apdft.conf.template") as fh:
    template = jinja.Template(fh.read())

  env = {}
  env["deltaZ"] = deltaZ
  env["deltaR"] = deltaR

  return template.render(**env)


def copy_commands(order1, order2):
  # basedir = os.path.dirname(os.path.abspath(__file__))
  basedir = "/Users/takafumishiraogawa/Program/APDFT/test/TStest/num_stab_n2/develop_code"
  # copyfile = "%s/template/commands.sh" % basedir
  copyfile = "template/commands.sh"
  copy_directry = "QM/delta-%s-%s" % (str(order1), str(order2))
  shutil.copy(copyfile, copy_directry)

  return


# If QM/ exists, it is removed.
if os.path.isdir("QM/"):
  shutil.rmtree("QM/")

# Set parameters.
change_frac_num = 5.0 * (10.0 ** (-8))
eval_num = 7
# That is, the number of evaluations is 140, and
# the range is from 10 ** (-7) to 10 ** 0.

delta = 0.0
count = 0

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
    inputfile = gener_inputs(delta, delta)
    with open("%s/apdft.conf" % path, "w") as inp:
        inp.write(inputfile)

    # Copy commands.sh
    copy_commands(i + 1, j + 1)
