import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 
import os
import subprocess as sub
import time
from numba import jit, njit, cuda
sub.call("rm -rf *.dat",shell=True)

#simulation parameters organized my use or type:
#**************************************************************
#The user will create different physical scenarios with these.
#macostate:
N=20000 #number of particles
rho=0.8 #density
E=-3.00 #total energy per particle
#time parameters:
nblk=5
nstep=10
dt=0.001
#**************************************************************
#probably most of this stuff can be put into initiaization function
#The user does not need to touch these parameters ordinarily.
#boundary conditions:
vol=N/rho
L=vol**(1/3)

#chemical/physical:
epsilon=1.0
sigma=1.0
#approximations
rcut=2.5*sigma
vcut=4.0*epsilon*((sigma/rcut)**12-(sigma/rcut)**6)
#observables:
n_obs=4 #Number of properties of the "walker"
iv = 0 #Potential energy index
iw = 1 #Virial index
it = 2 #kinetic energy index 
ie = 3 #Total energy index
n_props=n_obs
#for g(r)
igofr = n_props
nbins = 1
n_props = n_props + nbins
bin_size = (L/2.0)/nbins
#for C_v(t_d)
i_cv = n_props
t_delay=10
n_props = n_props + t_delay
#for C_pq(t_d)
#is there any reason why there would be two different delay times
i_pq=n_props
nq=10 #number of wave vectors
#can ask user to write file containing components of the q's 
#This is our first simple version for defining q along a single axis in q space
qvec=np.empty(nq)
for i in range(nq):
  qvec[i]=i*2*np.pi/L

n_props = n_props + t_delay*nq
blk_norm=0.0

walker=np.zeros(n_props)
blk_av=np.zeros(n_props)
exp_av=np.zeros(n_props)
std2=np.zeros(n_props)

#**************************************************************
#setup initial conditions
def initialize():
  #Global Variabless
  global state

  #This part computes nced and the number of vacancies in the lattice
  count=0
  while count < N:
    N_test=4*(count**3)
    N_test_1=4*(count+1)**3
    if N == N_test:
      hole_flag=1
      nced=count
      N_hole=0
      break
    elif (N_test<N) and (N<N_test_1):
      hole_flag=2
      nced=count+1
      N_hole=N_test_1-N
    count+=1


  #microstate:
  size1=N+N_hole
  state = np.zeros((size1, 2, 3))
  


  #initial positions
  #array of positions of 4 particles associated with front,lower,left
  #corner of FCC unit cell 
  qfcc=np.zeros((4,3))
  dimc=L/nced   #sidelength of unit cell
  ncells=nced**3 #number of unit cells in PBC box

  qfcc[0][0] = 0.0
  qfcc[0][1] = 0.0
  qfcc[0][2] = 0.0

  qfcc[1][0] = 0.5
  qfcc[1][1] = 0.5
  qfcc[1][2] = 0.0

  qfcc[2][0] = 0.0
  qfcc[2][1] = 0.5
  qfcc[2][2] = 0.5

  qfcc[3][0] = 0.5
  qfcc[3][1] = 0.0
  qfcc[3][2] = 0.5

  #translate copies of positions in array to fill up entire PBC cell
  #with FCC lattice points
  k=0
  for x_iter in range(nced):
    rel_x = x_iter*dimc
    for y_iter in range(nced):
      rel_y = y_iter*dimc
      for z_iter in range(nced):
        rel_z = z_iter*dimc
        for j in range(4):
          #perform the translation and place a particle there if
          #we have still not used up all the particles
          #changed N to N+N_hole, or else you will not start 
          #with full lattice to subtract off from
          #for no vacancies, N_hole==0
          if k<(size1):
            state[k][0][0] = rel_x + qfcc[j][0]*dimc
            state[k][0][1] = rel_y + qfcc[j][1]*dimc
            state[k][0][2] = rel_z + qfcc[j][2]*dimc
            #make PBC centered at origin
            state[k][0][0]+= - L*int(np.round(state[k][0][0]/L))
            state[k][0][1]+= - L*int(np.round(state[k][0][0]/L))
            state[k][0][2]+= - L*int(np.round(state[k][0][0]/L))
          k+=1

  #This part handles case when N_hole != 0.
  #We randomly delete N_hole many particles from full lattice
  #where N+N_hole is the closest "magic number" larger than N
  if hole_flag == 2:
    rnd_index=random.sample(range(size1), N_hole)
    state = np.delete(state, rnd_index, axis = 0)
  #initial velocities
  #calculate total potential energy from initial positions using LJV
  # start = time.time()
  vtot= np.zeros(1)
  # for particle_i in range(N-1):
  #   for particle_j in range(particle_i+1,N):
  #     vtot = np.add(vtot,LJV(state,particle_i,particle_j))
  # print("Outer loop ran in: " + str(time.time() - start))
  
  start = time.time()
  vtot += LJV2(state)
  print("Inner loop ran in: " + str(time.time() - start))
  #this is a warning that something is unphysical because
  #by definintion ekin>0.0, always
  ekin=(E*N)-vtot
  if ekin<0.0:
    print("WARNING: Unphysical initial configuration.")
  
  #use thermodynamics formula <K>=3T/2 to calculate temperature
  T=(2.0/3.0)*(ekin/N)

  #boost reference frame to center of mass frame of particle cloud
  vnet=np.zeros(3)
  for i in range(N):
    #uniform random velocities (adjusted later)
    state[i][1]=np.random.uniform(low=-0.5, high=0.5,size= 3)

    #net velocity components
    vnet = np.add(vnet, state[i][1])
    #vnet[0]=vnet[0]+vx[i]

  
  #components of net velocity per particle
  vnet = np.divide(vnet,N)
  #vnet[0]=vnet[0]/N

  #subtract off net velocity of cloud from velocity of each particle
  for i in range(N):
    state[i][1] = np.subtract(state[i][1], vnet)
    #vx[i]=vx[i]-vnet[0]

  #calculate velocities using the thermo formula
  sumv2= np.zeros(3)
  for i in range(N):
    sumv2 = np.add(sumv2, np.sum(np.square(state[i][1])))
  sumv2 = np.divide(np.sum(sumv2), N)
  
  fs = (3*T/sumv2)**0.5
  #rescale velocities so that they satisfy the thermo formula
  for i in range(N):
    state[i][1] = np.multiply(state[i][1], fs)
    #vx[i] *= fs

  print("(N,V,E) = ", N," ", vol," ", E)
  print("PBC cell side length: L = ", L)
  print("Total potential energy: epot = ", vtot)
  print("Total energy: E = ", E)
  print("Temperature: T = ", T)

#**************************************************************
#potential energy for a pair of particles function
#LJV(particles x positions, particles y positions, particles z positions, 1st particle index, 2nd particle index)
def LJV(state, particle_i,particle_j):
  temp_pos = state[particle_i][0] - state[particle_j][0]
  # dx0=q1[idi]-q1[idj]
  temp_pos = np.subtract(temp_pos, np.multiply(np.round(np.divide(temp_pos, L)), L))
  #dx0=dx0 - L*int(round(dx0/L))
  dr = np.sqrt(np.sum(np.square(temp_pos)))
  #dr=dx0**2+dy0**2+dz0**2
  #dr=dr**0.5
  v=4.0*epsilon*((sigma/dr)**12-(sigma/dr)**6)
  if dr>=rcut:
    v=0.0
  if dr<rcut:
    v=v-vcut
  return v
@jit(nopython=True)
def LJV2(state):
  vtot= np.zeros(1)
  for particle_i in range(N-1):
    for particle_j in range(particle_i+1,N):
        temp_pos = state[particle_i][0] - state[particle_j][0]
        # dx0=q1[idi]-q1[idj]
        x = np.divide(temp_pos, L)
        x = np.round(x[0])
        temp_pos = np.subtract(temp_pos, np.multiply(x, L))
        #dx0=dx0 - L*int(round(dx0/L))
        dr = np.sqrt(np.sum(np.square(temp_pos)))
        #dr=dx0**2+dy0**2+dz0**2
        #dr=dr**0.5
        v=4.0*epsilon*((sigma/dr)**12-(sigma/dr)**6)
        if dr>=rcut:
          v=0.0
        if dr<rcut:
          v=v-vcut
        vtot = np.add(vtot, v)
  return vtot
#**************************************************************
#velocity Verlet move function
def move():
  f0x=np.zeros(N)
  f0y=np.zeros(N)
  f0z=np.zeros(N)
  
  fx=np.zeros(N)
  fy=np.zeros(N)
  fz=np.zeros(N)

  for i in range(N):
    f0x[i]=force(i,0)
    f0y[i]=force(i,1)
    f0z[i]=force(i,2)

  for i in range(N):
    x[i]=x[i]+vx[i]*dt+0.5*f0x[i]*(dt**2)
    y[i]=y[i]+vy[i]*dt+0.5*f0y[i]*(dt**2)
    z[i]=z[i]+vz[i]*dt+0.5*f0z[i]*(dt**2)

  for i in range(N):
    fx[i]=force(i,0)
    fy[i]=force(i,1)
    fz[i]=force(i,2)

  for i in range(N):
    vx[i]=vx[i]+0.5*(fx[i]+f0x[i])*dt
    vy[i]=vy[i]+0.5*(fy[i]+f0y[i])*dt
    vz[i]=vz[i]+0.5*(fz[i]+f0z[i])*dt
  
  for i in range(N):
    x[i]+= - L*int(round(x[i]/L))
    y[i]+= - L*int(round(y[i]/L))
    z[i]+= - L*int(round(z[i]/L))

#**************************************************************
#force function
def force(idi, idim):
  fdim=0.0
  rsep=np.zeros(3)
  for idj in range(N):
    if idi!=idj:
      rsep[0]=x[idi]-x[idj]
      rsep[0]+= -L*int(round(rsep[0]/L))
      rsep[1]=y[idi]-y[idj]
      rsep[1]+= -L*int(round(rsep[1]/L))
      rsep[2]=z[idi]-z[idj]
      rsep[2]+= -L*int(round(rsep[2]/L))
      dr=rsep[0]**2+rsep[1]**2+rsep[2]**2
      dr=dr**0.5
      if dr<rcut:
        c=epsilon*(48.0*(sigma**12/dr**14) - 24.0*(sigma**6/dr**8))     
        fdim+=c*rsep[idim]
  return fdim
      
#**************************************************************
#measures observables
def measure():
  v = 0.0
  w = 0.0
  t = 0.0
  #reset the hystogram of g(r)
  for i in range(igofr, igofr+nbins):
    walker[i]=0.0
  for i in range(N-1):
    for j in range(i+1,N):
      dx = x[i] - x[j]
      dx += -L*int(round(dx/L))
      dy = y[i] - y[j]
      dy += -L*int(round(dy/L))
      dz = z[i] - z[j]
      dz += -L*int(round(dz/L))
      dr = dx**2 + dy**2 + dz**2
      dr = dr**0.5

      #g(r)
      bin = igofr + int(dr/bin_size)
      if bin < (igofr + nbins):
        walker[bin] += 2.0
     
      if dr < rcut:
        vij = 4.0*epsilon*((sigma/dr)**12-(sigma/dr)**6) - vcut 
        wij = 1.0*((sigma/dr)**12) - 0.5*((sigma/dr)**6)

        #Potential energy
        v += vij
        #Virial 
        w += wij
        #vshift += vcut

  #Kinetic energy
  for i in range(N):
    t += 0.5*(vx[i]**2 + vy[i]**2 + vz[i]**2)
 
  walker[iv] = v
  walker[iw] = 48.0*w/3.0
  walker[it] = t
  walker[ie] = walker[iv] + walker[it]

  #instantaneous positions and velocities
  pos=open("positions.dat",'a')
  vel=open("velocities.dat",'a')
  
  for i in range(N):
    pos.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(x[i],y[i],z[i]))
    vel.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(vx[i],vy[i],vz[i]))
  
  pos.close()
  vel.close()

#**************************************************************
#main
#interface()
initialize()
for iblk in range(nblk):
  average(1)
  for t in range(nstep):
    for m in range(t_delay):
      measure_dyn(m)
      move()
    measure()
    average(2)
    print("t = ",t)
  average(3)
#**************************************************************
