import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import sys 

N=int(sys.argv[1])
L=float(sys.argv[2])

#N=108
#L=

x,y,z=np.loadtxt("positions.dat",usecols=(0,1,2),unpack=True)

nsteps=len(x)/N
nsteps=int(nsteps)

xti=np.zeros((N,nsteps))
yti=np.zeros((N,nsteps))
zti=np.zeros((N,nsteps))

for i in range(N):
  for t in range(nsteps):
     s=N*t+i
     xti[i][t]=x[s]
     yti[i][t]=y[s]
     zti[i][t]=z[s]


#print(x)

#print(xti)


fig = plt.figure()
ax = Axes3D(fig)

def animate(t):
    ax.clear()
    ax.set_xlim(-L/2.0,L/2.0)
    ax.set_ylim(-L/2.0,L/2.0)
    ax.set_zlim(-L/2.0,L/2.0)

    for i in range(N):
        xp=xti[i][t]
        yp=yti[i][t]
        zp=zti[i][t]
        ax.scatter(xp,yp,zp)
#        patch=patches.Circle((xp,yp), 0.01, color='blue')
#        ax.add_patch(patch)

anim=animation.FuncAnimation(fig,animate,nsteps,interval=1, blit=False)

plt.show()


#anim.save('md_test_1.MP4', writer='imagemagick', fps=20)
