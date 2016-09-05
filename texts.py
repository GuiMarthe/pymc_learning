# texts.py
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


src = 'book/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/Chapter1_Introduction/data/txtdata.csv'

data = np.loadtxt(src)


n_data = len(data)

print(n_data)
print(data)

#ploting
plt.figure(1)
plt.subplot(211)
plt.bar(np.arange(n_data), data, color = '#388ABD')
plt.xlabel("Time(days)")
plt.ylabel("Txt messages recieved")
plt.title("texting habbits: did they change?")
plt.xlim(0, n_data)

plt.subplot(212)
plt.hist(data)
plt.show()
