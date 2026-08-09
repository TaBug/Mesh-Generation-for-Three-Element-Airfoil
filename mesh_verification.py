import numpy as np
from plotgri import readgri
from matrices_generator import getI2E, getB2E, edgehash, area


def meshTest(griFile):
    Mesh = readgri(griFile)
    E = Mesh['E']
    I2E = getI2E(griFile, False)
    B2E = getB2E(griFile, False)
    In, Bn, lIn, lBn = edgehash(griFile, False)
    Error = np.zeros([len(E), 2])
    for i in range(len(In)):
        elemL = I2E[i][0] - 1
        elemR = I2E[i][2] - 1
        Error[elemL] += In[i] * lIn[i]
        Error[elemR] -= In[i] * lIn[i]

    for i in range(len(Bn)):
        elem = B2E[i][0] - 1
        Error[elem] += Bn[i] * lBn[i]
    return Error


def main():
    ETest = meshTest('gri/test.gri')
    print(f'The maximum magnitude of the error for test mesh = {np.max(abs(ETest))}')
    EAll = meshTest('gri/all.gri')
    print(f'The maximum magnitude of the error for coarse mesh = {np.max(abs(EAll))}')
    EAllTrail = meshTest('gri/smoothed_local_all_trail.gri')
    print(f'The maximum magnitude of the error for coarse mesh = {np.max(abs(EAllTrail))}')
    EAllAll = meshTest('gri/refined_local_all.gri')
    print(f'The maximum magnitude of the error for coarse mesh = {np.max(abs(EAllAll))}')


if __name__ == "__main__":
    main()
