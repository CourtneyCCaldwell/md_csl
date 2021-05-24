import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

rad,data1,err1=np.loadtxt('gofr_statistics.dat',usecols=(0,1,2),unpack=True)
tds,data2,err2=np.loadtxt('vacf_statistics.dat',usecols=(0,1,2),unpack=True)
data3,err3=np.loadtxt('ddcf_statistics.dat',usecols=(0,1),unpack=True)

plt.errorbar(rad,data1,yerr=err1,ecolor='r',capsize=1.0,ls='--', label = "g(r)")

#plt.errorbar(tds,data2,yerr=err2,ecolor='r',capsize=1.0,ls='--',label = "C_v(t_d)")
#plt.axhline(0.0,c='g')

#data3_sort=np.empty((tds,nq))
#data3_sort=np.reshape(data3, (tds, nq))
#ax = sns.heatmap(data3_sort)
#ax.set_title('p(q,t_d)')
#ax.set_xlabel('t_d')
#ax.set_ylabel('nq')


plt.legend()
plt.show()



