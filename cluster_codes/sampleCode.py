import numpy as np
import sys

l1, l2 = sys.argv[1].split(' ')#input('lengths: ').split(' ')
area = float(l1) * float(l2)
#area = np.ones(3)
fName = 'data_' + str(l1) + str(l2) + '.txt'

np.savetxt(fName, [area],  fmt = '%f')
