import os
import glob
import shutil

# input
field_strength = "0.001"

dir_fd = "./finite_difference/"
if os.path.isdir(dir_fd):
  shutil.rmtree(dir_fd)

# Get all file names in ./template/
files = glob.glob('./template/*')

# Exception handling
inputs = ['apdft.conf', 'imp_mod_cli1.sh', 'imp_mod_cli2.sh', 'mol.xyz']
# for i, file in enumerate(files):
#   if file not in inputs:
#     NotImplemented("template/ is not valid.")

# Read apdft.conf
f = open('./template/apdft.conf', 'r')
data = f.readlines()
f.close()

field_param_list = []
field_param_list.append("%s, 0.0, 0.0" % field_strength)
field_param_list.append("-%s, 0.0, 0.0" % field_strength)
field_param_list.append("0.0, %s, 0.0" % field_strength)
field_param_list.append("0.0, -%s, 0.0" % field_strength)
field_param_list.append("0.0, 0.0, %s" % field_strength)
field_param_list.append("0.0, 0.0, -%s" % field_strength)

# Set parameters
flag_string = 'apdft_finite_field = True'
field_param = []
field_param_string = 'apdft_field_vector = '
for i in range(6):
  field_param.append("%s%s" % (field_param_string, field_param_list[i]))

# Set directories for calculations
os.mkdir('finite_difference/')
field_direc = []
field_direc.append('./finite_difference/x_+/')
field_direc.append('./finite_difference/x_-/')
field_direc.append('./finite_difference/y_+/')
field_direc.append('./finite_difference/y_-/')
field_direc.append('./finite_difference/z_+/')
field_direc.append('./finite_difference/z_-/')

# Write apdft.conf (original + inputs for the field)
for i in range(6):
  os.mkdir(field_direc[i])

# Copy files other than apdft.conf
for i in range(6):
  shutil.copyfile("./template/%s" % inputs[1], "%s%s" % (field_direc[i], inputs[1]))
  shutil.copyfile("./template/%s" % inputs[2], "%s%s" % (field_direc[i], inputs[2]))
  shutil.copyfile("./template/%s" % inputs[3], "%s%s" % (field_direc[i], inputs[3]))

# Generate apdft.conf
for i in range(6):
  with open("%s%s" % (field_direc[i], 'apdft.conf'), mode='w') as apdft_conf_out:
    for j, line in enumerate(data):
      print(line.replace('\n', ''), file=apdft_conf_out)
      if j == 1:
        print(flag_string, file=apdft_conf_out)
        print(field_param[i], file=apdft_conf_out)

# Script for stating all the calculations
with open("run_all.sh", mode='w') as run_all:
  for i in range(6):
    print("cd %s" % (field_direc[i]), file=run_all)
    print(". imp_mod_cli1.sh", file=run_all)
    print("div_QM.py 8", file=run_all)
    print(". imp_mod_cli2.sh", file=run_all)
    print("cd ../../", file=run_all)