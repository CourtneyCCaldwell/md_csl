import numpy as np
import matplotlib.pyplot as plt


x1=np.loadtxt("ekinetic.dat",usecols=(0),unpack=True)
x2=np.loadtxt("epotential.dat",usecols=(0),unpack=True)
x3=np.loadtxt("etotal.dat",usecols=(0),unpack=True)
x4=np.loadtxt("gofr.dat",usecols=(0),unpack=True)
x5=np.loadtxt("vacf.dat",usecols=(0),unpack=True)

#print(np.mean(x1[:]))

#print(x1)

#plt.plot(x1,label="kinetic")
#plt.plot(x2,label="potential")
#plt.plot(x3,label="total")
plt.plot(x4,label="g(f)")
#plt.plot(x5,label="C_v(t_d)")

plt.legend()
plt.show()

#plt.axhline(0.0)
          
