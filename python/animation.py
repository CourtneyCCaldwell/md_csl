import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import sys 

#size of particles in animation
size = 100.0

x,y,z=np.loadtxt("positions.dat",usecols=(0,1,2),unpack=True)
anim_param  = np.loadtxt("sim_params.in", unpack=True)

N = int(anim_param[0])
rho = anim_param[1]
L = (N/rho)**(1.0/3.0)

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
        ax.scatter(xp,yp,zp,s=size)

anim=animation.FuncAnimation(fig,animate,nsteps,interval=1, blit=False)
plt.show()
writergif = animation.PillowWriter(fps=10)
anim.save('CSL_animation.gif', writer=writergif)
