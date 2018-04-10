import matplotlib.pyplot as plt
import pickle
import os,sys

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)


plot_range = [1,600]
x = range(plot_range[0],plot_range[1])

y=[]
with open('record.pkl','rb') as f:
    record = pickle.load(f)
for idx in x:
    y.append(record[idx])
plt.plot(x,y)

plt.show()