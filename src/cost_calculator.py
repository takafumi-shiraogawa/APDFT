import math
import itertools as it

num_sites = 20
limit_num_sites = 20

target_atom_positions = list(range(num_sites))

count_proc_mol = 0

for num_mut_atoms in range(limit_num_sites + 1):
  if num_mut_atoms % 2 != 0:
    continue

  for mut_atom_positions in it.combinations(target_atom_positions, num_mut_atoms):
    positions_mut_atom_positions = []
    for i in range(num_mut_atoms):
      positions_mut_atom_positions.append(i)

    for pos_positions_mut_atom_positions in it.combinations(positions_mut_atom_positions, int(num_mut_atoms / 2)):
      count_proc_mol += 1
print(count_proc_mol)

cost2 = 0
for i in range(limit_num_sites + 1):
  if i % 2 != 0:
    continue
  half = i / 2
  half = int(half)
  pos = math.factorial(num_sites) // (math.factorial(half) * math.factorial(num_sites - half))
  neg = math.factorial(num_sites - half) // (math.factorial(half) * math.factorial(num_sites - half - half))
  cost2 += pos * neg
print(cost2)