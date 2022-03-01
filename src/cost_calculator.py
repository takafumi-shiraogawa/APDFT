import math

num_sites = 14
num_muts = 2

cost = 1

for i in range(5):
  factor = i + 1
  cost += math.factorial(num_sites) // math.factorial(num_sites - factor * num_muts)

print(cost)