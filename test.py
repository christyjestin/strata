import numpy as np

val = np.array([0,0,0])

for i in range(100000):

    val = val+ (np.random.rand(3) - 0.5)

print(val/100000)