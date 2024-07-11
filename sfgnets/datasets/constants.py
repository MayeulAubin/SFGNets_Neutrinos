import numpy as np

RANGES = np.array([[ -985.92 ,   985.92 ],
                   [ -257.56 ,   317.56 ],
                   [-2888.776,  -999.096]])  # detector ranges (X, Y, Z)

CUBE_SIZE = 10.27 / 2  # half of cube size in mm

ORIGIN = RANGES.mean(axis=1)