#!/usr/bin/env python

# A code for analyzing APDFT results with different fractional numbers
# for finite differential in the perturbed electron density
# after running gener_apdft_conf.py

# Requirements:
#   order.inp
#   QM/

import csv
import matplotlib.pyplot as plt
import numpy as np


def get_target_value(target_var, target, dict_inp, apdft_order):
  for i, row in enumerate(dict_inp):
    if target_var == "Z":
      if i == 0:
        return row["%s%s" % (target, str(apdft_order))]
    else:
      return row["%s%s" % (target, str(apdft_order))]


# Read specified variables dR or dZ
order_inp = open('order.inp', 'r')
specified_var = order_inp.read()
if specified_var != "Z" and specified_var != "R":
  raise ValueError("Error! order.inp should be Z or R!")

eval_num = 7
div_num = 20
count = 0
apdft_order = 3

reference_contributions = np.zeros((eval_num, div_num, apdft_order))
target_contributions = np.zeros((eval_num, div_num, apdft_order))
total_contributions = np.zeros((eval_num, div_num, apdft_order))

for i in range(eval_num):
  for j in range(div_num):
    count += 1

    path = "QM/delta-%s-%s" % (i + 1, j + 1)

    # inp_reference = open("%s/reference_contributions.csv" % path, "r")
    # inp_target = open("%s/target_contributions.csv" % path, "r")
    # inp_total = open("%s/total_contributions.csv" % path, "r")

    # dict_inp_reference = csv.DictReader(inp_reference)
    # dict_inp_target = csv.DictReader(inp_target)
    # dict_inp_total = csv.DictReader(inp_total)

    for k in range(apdft_order):
      inp_reference = open("%s/reference_contributions.csv" % path, "r")
      inp_target = open("%s/target_contributions.csv" % path, "r")
      inp_total = open("%s/total_contributions.csv" % path, "r")

      dict_inp_reference = csv.DictReader(inp_reference)
      dict_inp_target = csv.DictReader(inp_target)
      dict_inp_total = csv.DictReader(inp_total)
      reference_contributions[i, j, k] = get_target_value(specified_var,
        "reference_contributions_order", dict_inp_reference, k)
      target_contributions[i, j, k] = get_target_value(specified_var,
        "target_contributions_order", dict_inp_target, k)
      total_contributions[i, j, k] = get_target_value(specified_var,
        "total_contributions_order", dict_inp_total, k)
      # inp_reference.see
      inp_reference.close()
      inp_target.close()
      inp_total.close()

    # inp_reference.close()
    # inp_target.close()
    # inp_total.close()


# Prepare one-dimensional data for plot
count = -1

change_frac_num = 5.0 * (10.0 ** (-8))

# frac_num = np.zeros(eval_num * div_num)
frac_num = np.zeros(div_num + (eval_num - 1) * (div_num - 2) - 1)
reference_contributions_data = np.zeros(
    (div_num + (eval_num - 1) * (div_num - 2) - 1, apdft_order))
target_contributions_data = np.zeros(
    (div_num + (eval_num - 1) * (div_num - 2) - 1, apdft_order))
total_contributions_data = np.zeros(
    (div_num + (eval_num - 1) * (div_num - 2) - 1, apdft_order))

for i in range(eval_num):
  effect_change_frac_num = change_frac_num * (10.0 ** i)

  # Avoid duplications
  for j in range(div_num):

    if i > 0:
      # It is an ad hoc treatment for eval_num.
      if j == 0 or j == 1:
        continue

    if i == 5 and j == 9:
      continue

    count += 1
    delta = effect_change_frac_num * (j + 1)
    print(count, delta)
    if specified_var == "Z":
      frac_num[count] = delta
    else:
      frac_num[count] = delta * 0.1

    for k in range(apdft_order):
      reference_contributions_data[count, k] = reference_contributions[i, j, k]
      target_contributions_data[count, k] = target_contributions[i, j, k]
      total_contributions_data[count, k] = total_contributions[i, j, k]


# Get deviations from the standard value
devi_reference_contributions_data = np.zeros(
    (div_num + (eval_num - 1) * (div_num - 2) - 1, apdft_order))
devi_target_contributions_data = np.zeros(
    (div_num + (eval_num - 1) * (div_num - 2) - 1, apdft_order))
devi_total_contributions_data = np.zeros(
    (div_num + (eval_num - 1) * (div_num - 2) - 1, apdft_order))
count = -1
for i in range(eval_num):
  for j in range(div_num):

    if i > 0:
      # It is an ad hoc treatment for eval_num.
      if j == 0 or j == 1:
        continue

    if i == 5 and j == 9:
      continue

    count += 1

    for k in range(apdft_order):

      # Here 5, 9 specifies contributions of 0,05 for dZ and.0.005 for dR.
      devi_reference_contributions_data[count, k] = \
          abs(reference_contributions_data[count, k] -
              reference_contributions[5, 9, k])
      devi_target_contributions_data[count, k] = \
          abs(target_contributions_data[count, k] -
              target_contributions[5, 9, k])
      devi_total_contributions_data[count, k] = \
          abs(total_contributions_data[count, k] -
              total_contributions[5, 9, k])


# Save CSV files
np.savetxt('frac_num.dat.csv', frac_num)
np.savetxt('reference_contributions.dat.csv',
           reference_contributions_data, delimiter=',')
np.savetxt('target_contributions.dat.csv',
           target_contributions_data, delimiter=',')
np.savetxt('total_contributions.dat.csv',
           total_contributions_data, delimiter=',')
# Deviations
np.savetxt('devi_reference_contributions.dat.csv',
           devi_reference_contributions_data, delimiter=',')
np.savetxt('devi_target_contributions.dat.csv',
           devi_target_contributions_data, delimiter=',')
np.savetxt('devi_total_contributions.dat.csv',
           devi_total_contributions_data, delimiter=',')

# Plot data
for i in range(apdft_order):
  plt.plot(frac_num[:], reference_contributions_data[:, i])
  plt.xscale("log")
  plt.savefig("reference_contributions_%s.png" % str(i + 1))
  plt.close('all')

  plt.plot(frac_num[:], target_contributions_data[:, i])
  plt.xscale("log")
  plt.savefig("target_contributions_%s.png" % str(i + 1))
  plt.close('all')

  plt.plot(frac_num[:], total_contributions_data[:, i])
  plt.xscale("log")
  plt.savefig("total_contributions_%s.png" % str(i + 1))
  plt.close('all')

  plt.plot(frac_num[:], devi_reference_contributions_data[:, i])
  plt.xscale("log")
  plt.yscale("log")
  plt.savefig("devi_reference_contributions_%s.png" % str(i + 1))
  plt.close('all')

  plt.plot(frac_num[:], devi_target_contributions_data[:, i])
  plt.xscale("log")
  plt.yscale("log")
  plt.savefig("devi_target_contributions_%s.png" % str(i + 1))
  plt.close('all')

  plt.plot(frac_num[:], devi_total_contributions_data[:, i])
  plt.xscale("log")
  plt.yscale("log")
  plt.savefig("devi_total_contributions_%s.png" % str(i + 1))
  plt.close('all')


# Plot data for a paper
# Set parameters for the figure
if specified_var == "Z":
  xaxis_name = "δZ"
  yaxis_name = "E(δZ) - E(0.05)"
else:
  xaxis_name = "δR"
  yaxis_name = "E(δR) - E(0.005)"

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams["xtick.major.size"] = 8
plt.rcParams["ytick.major.size"] = 8

fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

ax.plot(frac_num[:], devi_total_contributions_data[:, 1], label='APDFT2')
ax.plot(frac_num[:], devi_total_contributions_data[:, 2], label='APDFT3')

fig.show()

xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()
xlabel = xaxis_name
ylabel = yaxis_name

ax.set_xticklabels(xticklabels, fontsize=18, fontname='Arial')
ax.set_yticklabels(yticklabels, fontsize=18, fontname='Arial')
ax.tick_params(pad=7)
# ax.set_xlabel('$\it{xlabel}$', fontsize=22, fontname='Arial')
# ax.set_ylabel('$\it{ylabel}$', fontsize=22, fontname='Arial')
if specified_var == "Z":
  ax.set_xlabel('$\it{δZ}$', fontsize=20, fontname='Arial')
  ax.set_ylabel('$\it{|E(δZ) - E(0.05)|}$', fontsize=20, fontname='Arial')
else:
  ax.set_xlabel('$\it{δR}$ / Å', fontsize=20, fontname='Arial')
  ax.set_ylabel('$\it{|E(δR) - E(0.005)|}$', fontsize=20,
                fontname='Arial')

ax.legend(loc='upper right', fontsize=18)
# ax.grid()

plt.xscale("log")
plt.yscale("log")

if specified_var == "Z":
  plt.xlim(10 ** (-7), 10)
  plt.xticks([10 ** (-7), 10 ** (-5), 10 ** (-3), 10 ** (-1), 10])

  ymin = 10 ** (-10)
  ymax = 10 ** 2
  plt.ylim(ymin, ymax)
  plt.yticks([ymin, 10 ** (-6), 10 ** (-4), 10 ** (-2), 10 ** (0), ymax])
  plt.plot([0.05, 0.05], [ymin, ymax], "grey")

else:
  plt.xlim(10 ** (-8), 0.1)
  plt.xticks([10 ** (-8), 10 ** (-6), 10 ** (-4), 10 ** (-2), 1])

  ymin = 10 ** (-4)
  ymax = 10 ** 8
  plt.ylim(ymin, ymax)
  plt.yticks([ymin, 10 ** (-2), 10 ** (0), 10 ** 2, 10 ** 4, 10 ** 6, ymax])
  plt.plot([0.005, 0.005], [ymin, ymax], "grey")

plt.savefig("devi_target_contributions_%s_N2.tiff" % specified_var, dpi=600)
plt.close('all')
