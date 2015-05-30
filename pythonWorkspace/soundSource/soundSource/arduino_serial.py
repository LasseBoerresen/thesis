import serial
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt



ser = serial.Serial('/dev/ttyUSB0', 9600)

rec = np.ndarray([8000])


for i in range(8000):
    print ser.readline()
    rec[i] = int(ser.readline())