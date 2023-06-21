import random 
import matplotlib.pyplot as plt


a = [random.uniform(1,100) for i in range(10000)]


plt.hist(a)
plt.show()