#1. finish statistics file
#2. add user friendly plotting file
#3. add user friendly animation file
#4. add calculation of standard errors 
#5. clean up, make tidy
#6. add lots of comments
#7. fix animation file
#8. go through a second time to check I understand everything

import numpy as np
import cmath
import random
import subprocess as sub
sub.call("rm -rf *.dat",shell=True)

#***************************************************************************************************************************************
#These variables can be changed by the user to do different simulates for different possible realizations of the canonical ensemble. 
#***************************************************************************************************************************************

#load from simulation parameters file sim_params.in
p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = np.loadtxt("sim_params.in", usecols = 0,unpack=True)

#ENSEMBLE PARAMETERS: The user can simulate different macrostates by choosing different values for the parameters of the
#microcanonical ensemble (N,V,E).

nPart = int(p1) #The number of particles, N.
rho = p2 #The density. This is related to the volume V.
eTot = p3 #The total energy per particle. This is related to the total energy E.

#TIME PARAMETERS: These control the quality of the simulation evolution and the quality of the statistics.

nBlock = int(p4) #Number of blocks. This controls number of measurements that are used to calculate the statistics for the oberservables.
nStep =  int(p5) #Number of time steps in a block. The simulated system changes its microstate from one time step to another.
dt = p6  #Time step length. This is the approximation to the time differential. It is small but necessarily finite, because the computer 
         #does "know" calculus, and can only do basic arthimetic.

#***************************************************************************************************************************************

#BOUNDARY CONDITIONS: These are related to the overal geometry and size of the system.

Vol = nPart/rho   #The system's volume. This is computed from the number of particles and the denisty.
lBox = Vol**(1/3) #The side length of the system. We assume the volume is a cube and compute the length of cube's edges from the volume.

#CHEMICAL PARAMETERS: These are related to the properties of the Lennard-Jones potential (LJV) energy function which determines 
#the interaction between different atoms.

epsilon = 1.0 #Is the minimum value of the LJP.
sigma = 1.0   #When r = sigma, the value of the LJP is zero. Sigma is also proportional to the minima of the potential.

#APPROXIMATIONS: Because the LJV drops off quickly with distance, we introduce a cut off distance beyond which the potential is 
#zero. Thus, the potential we actually use in the simulation is not the LJV, but rather a cut and shifted version of the LJV.

rCut = 2.5*sigma #The cut off distance. We write it as a multiple of the zero-value distance sigma.
vCut = 4.0*epsilon*((sigma/rCut)**12 - (sigma/rCut)**6) #The value of the potential evaluated at the cut off distance.
vShift = 0.0 #????This is issue we need to address, how do we not have to define as global inside functions????

#TAIL CORRECTIONS: These are variables are introduced to correct for errors introduced by the cut and shifted LJV. There are 
#theoretical arguments that justify the form of these expressions.

vTail = (8.0*np.pi*rho)/(9.0*pow(rCut,9)) - (8.0*np.pi*rho)/(3.0*pow(rCut,3)) #Related to the corrections to the potential energy
pTail = (32.0*np.pi*rho)/(9.0*pow(rCut,9)) - (16.0*np.pi*rho)/(3.0*pow(rCut,3)) #Related to corrections to the pressure

#STATIC OBSERVABLES: These variables are related to observables that do not change with time. Some are single numbers such as 
#pressure because the pressure of a system in equilibrium is a single number not a scalar field. Others such as g(r) are 
#functions of position. 

nOb = 4 #Number of indices reserved in the "walker" array for single number observables
iv = 0  #Potential energy index in walker 
iw = 1  #Virial index in walker
it = 2  #Kinetic energy index in walker
ie = 3  #Total energy index in walker
nProp = nOb #We keep updating the value of this variable as we add new function observables


igofr = nProp #Index in walker where the radial distribution function g(r) starts
nBin = int(p7) #Number of values g(r) is measured for
nProp = nProp + nBin #We keep updating the value of this variable as we add new function observables
sizeBin = (lBox/2.0)/nBin #The width in r of the bins used to compute g(r)

#DYNAMICAL OBSERVABLES: Observables that change as a function of time.

#for C_vv(t_d)
icvel = nProp #Index in walker where the velocity autocorrelation function C_vv(t_d) starts
nTdelay = int(p8) #Number of values C_vv(t_d) is measured for
nProp = nProp + nTdelay #We keep updating the value of this variable as we add new function observables

#for C_rhorho(t_d)
icrho = nProp #Index in walker where the density autocorrelation function C_rhorho(t_d) starts

nMomenta = int(p9) #Number of momenta wave vectos used to calculate C_rhorho(t_d)

#Array of the momenta. We need to define these, but for a fluid, which is isotropic, all the momenta we use point in the same direction, but have 
#different magnitudes. This is our first simple version for defining q along a single axis in q space.

qMomenta = np.empty(nMomenta) 
for i in range(nMomenta):
  qMomenta[i] = i*2*np.pi/lBox

nProp = nProp + nTdelay*nMomenta #We keep updating the value of this variable as we add new function observables

instFlag = p10 #flag that controls whether use want to save the instantaneous positions and velocities

#OBSERVABLES' STATISTICS:

blockNorm = 0.0 #The normalization factor that comes from the number of blocks
walker = np.zeros(nProp) #Array of all single number observables and function observables
blockAverage = np.zeros(nProp) #
experimentAverage = np.zeros(nProp) #

#***************************************************************************************************************************************

def initialize():
  """
  This function sets up the initial condition of the system. Given the number of particles provided by the user it tries its 
  best to arrange them in a face centered cubic (FCC) lattice, which is the prefered lattice of a frozen noble gas. It 
  assigns initial velocities to the particles using the energy provided by the user and the Maxwell-Boltzmann distribution.
  """

  #Global Variables
  global xPos,yPos,zPos,xVel,yVel,zVel,xVel0,yVel0,zVel0,qRho,qRho0

  #This part computes nUperlbox and the number of holes in the lattice. nUperlbox is the number of unit cells that fit along 
  #the edge of the PBC box of length lBox. 

  #It works by finding out whether npart is an FCC magic number or not. By magic number, we mean that the entire FCC lattice
  #(for a given number of unit FCC cells) is filled with no holes. If npart is not a magic number, this section of code 
  #determines how many holes in the FCC lattice there are.
 
  count = 0
  while count < nPart:       
    nTest = 4*(count**3)     #a magic number
    nTest1 = 4*(count+1)**3  #next highest magic number
    if nPart == nTest:       #checks if nPart is magic number
      holeFlag = 1           #flag if not a magic number
      nUperlbox = count      #number of unit cells needed to fit along edge of pbc cell
      nHole = 0              #no holes
      break
    elif (nTest<nPart) and (nPart<nTest1):  #loop until we find two magic numbers that such that nTest < nPart < nTest1
      holeFlag = 2           #flag if a magic number 
      nUperlbox = count + 1  #number of unit cells needed to fit along edge of pbc cell
      nHole = nTest1 - nPart #number of holes
    count += 1

  nObject = nPart + nHole    #number of particles plus number of holes
  #arrays of particle position components
  xPos = np.zeros(nObject)
  yPos = np.zeros(nObject)  
  zPos = np.zeros(nObject)
  #arrays of particle velocity components
  xVel = np.zeros(nPart)
  yVel = np.zeros(nPart)
  zVel = np.zeros(nPart)
  #v(t0) array for dynamical measurements
  xVel0 = np.zeros(nPart)  
  yVel0 = np.zeros(nPart)
  zVel0 = np.zeros(nPart)
  #qRho(t) and qRho0(t) arrays for dynamical measurements
  qRho = np.empty(nMomenta,dtype=np.complex_)
  qRho0 = np.empty(nMomenta,dtype=np.complex_)

  #initial positions
  #array of positions of 4 particles associated with front,lower,left corner of FCC unit cell 
  fccCell = np.zeros((4,3)) #array for unit cell positions
  lUcell = lBox/nUperlbox   #side length of unit cell
  nUcell = nUperlbox**3     #number of unit cells in PBC box
 
  #define unscaled positions for the 4 particle in the unit cell
  fccCell[0][0] = 0.0
  fccCell[0][1] = 0.0
  fccCell[0][2] = 0.0

  fccCell[1][0] = 0.5
  fccCell[1][1] = 0.5
  fccCell[1][2] = 0.0

  fccCell[2][0] = 0.0
  fccCell[2][1] = 0.5
  fccCell[2][2] = 0.5

  fccCell[3][0] = 0.5
  fccCell[3][1] = 0.0
  fccCell[3][2] = 0.5

  #Translate copies of positions in array to fill up entire PBC cell with FCC lattice points
  k = 0
  for ix in range(nUperlbox):
    for iy in range(nUperlbox):
      for iz in range(nUperlbox):
        for j in range(4):
          #Perform the translation and place a particle there if we have still not used up all the particles.
          #Note that we are placing more nObject particles not nPart. We will get rid of the appropriate number
          #of excess particles to get back down to nPart later.
          if k<(nObject):
            #Scale by lUcell the original unit cell positions and translate them along the various axes. Make PBC 
            #cell centered at origin by tranlating every particle position component by lBox/2. Otherwise all the
            #particle would be laid down in the first octant.
            xPos[k] = (ix + fccCell[j][0])*lUcell - (lBox/2)
            yPos[k] = (iy + fccCell[j][1])*lUcell - (lBox/2)
            zPos[k] = (iz + fccCell[j][2])*lUcell - (lBox/2)
          k+=1

  #This part handles the case when nHole != 0. We randomly select then delete nHole many particles from the full 
  #lattice where nObject=nPart+nHole is the closest "magic number" larger than nPart.
  if holeFlag == 2:
    iRandhole = random.sample(range(nObject), nHole) #generate array of randomly selected indices we want to delete
    xPos = np.delete(xPos, iRandhole) #get new arrays with only nPart many elements
    yPos = np.delete(yPos, iRandhole)
    zPos = np.delete(zPos, iRandhole)

  #This part initializes the  velocities. We calculate total potential energy from initial positions using LJV
  vtot = 0.0
  for i in range(nPart-1):
    for j in range(i+1,nPart):
      vpair = LJV(xPos,yPos,zPos,i,j)
      vtot += vpair

  #This is a warning that something is unphysical because by definition ekin>0.0, always.
  ekin = (eTot*nPart) - vtot
  if ekin<0.0:
    print("WARNING: Unphysical initial configuration.")

  #Use equipartition of energy for system of particles with no internal degrees of freedom, which imples K_total=3TN/2,
  #to calculate temperature.
  T = (2.0/3.0)*(ekin/nPart)

  #Boost reference frame to center of mass frame of particle cloud. This stops the particles from having an average 
  #velocity. 
  netVel = np.zeros(3)
  for i in range(nPart):
    #uniform random velocities (adjusted later)
    xVel[i] = np.random.uniform() - 0.5
    yVel[i] = np.random.uniform() - 0.5
    zVel[i] = np.random.uniform() - 0.5
    #net velocity components
    netVel[0] += xVel[i]
    netVel[1] += yVel[i]
    netVel[2] += zVel[i]

  #components of net velocity per particle
  netVel[0] /= nPart
  netVel[1] /= nPart
  netVel[2] /= nPart

  #subtract off net velocity of cloud from velocity of each particle
  for i in range(nPart):
    xVel[i] -= netVel[0]
    yVel[i] -= netVel[1]
    zVel[i] -= netVel[2]

  #calculate velocities using the thermo formula
  sumVelsqrd = 0.0
  for i in range(nPart):
    sumVelsqrd += xVel[i]**2 + yVel[i]**2 + zVel[i]**2
  sumVelsqrd /= nPart
  
  #rescale velocities so that they satisfy the thermo formula
  scaleVel = (3*T/sumVelsqrd)**0.5
  for i in range(nPart):
    xVel[i] *= scaleVel
    yVel[i] *= scaleVel
    zVel[i] *= scaleVel

  print("(N,V,E) = ", nPart," ", Vol," ", eTot)
  print("PBC cell side length: L = ", lBox)
  print("Total potential energy: epot = ", vtot)
  print("Total energy: E = ", eTot)
  print("Temperature: T = ", T)

#***************************************************************************************************************************************

#Interparticle distance function 
def partdist(xPart, yPart, zPart, iPart, jPart):
  """
  Calculates the distance between two particles. xPart, yPart, zPart are arrays of particle positions. iPart, jPart are the indices of
  the particles in the pair.
  """
  #NaN bug appears to come from calculation of particle distance for the initial configuration, so that the initial potential energy
  #can be calculated. But for some reason the initial potential energy gives a negative kinetic energy, even though in Dr. Vitali's
  #code this choice for the microcanon. ensemble should work.

  #the math problem is that the potential for the IC does not seem to be negative enough, need larger negative value
  
  xSep =  xPart[iPart] - xPart[jPart]
  xSep -= lBox*(int(round(xSep/lBox)))
  ySep =  yPart[iPart] - yPart[jPart]
  ySep -= lBox*(int(round(ySep/lBox)))
  zSep =  zPart[iPart] - zPart[jPart]
  zSep -= lBox*(int(round(zSep/lBox)))
  rSep =  (xSep**2 + ySep**2 + zSep**2)**0.5
  return rSep

#***************************************************************************************************************************************

def LJV(xPart1, yPart1, zPart1, iPart1, jPart1):
  """
  Calculates the potential energy of a pair of particles as a function of the separation distance. xPart, yPart, zPart are arrays of
  particle positions. iPart, jPart are the indices of the particles in the pair.
  """
  
  rSep1 = partdist(xPart1, yPart1, zPart1, iPart1, jPart1)
  v = 4.0*epsilon*((sigma/rSep1)**12 - (sigma/rSep1)**6)
  if rSep1>=rCut:
    v = 0.0
  if rSep1<rCut:
    v -= vCut #v is cut and shifted potential
  return v

#***************************************************************************************************************************************

#velocity Verlet move function
def move():
  """
  Performs the velocity Verlet algorithm.
  """

  #trial step components of all the forces of all the particles
  xForce0 = np.zeros(nPart)
  yForce0 = np.zeros(nPart)
  zForce0 = np.zeros(nPart)
  
  #components of all the forces of all the particles
  xForce = np.zeros(nPart)
  yForce = np.zeros(nPart)
  zForce = np.zeros(nPart)

  for i in range(nPart):
    xForce0[i] = force(i,0)
    yForce0[i] = force(i,1)
    zForce0[i] = force(i,2)
  
  #velocity verlet position formula
  for i in range(nPart):
    xPos[i] += xVel[i]*dt + 0.5*xForce0[i]*(dt**2)
    yPos[i] += yVel[i]*dt + 0.5*yForce0[i]*(dt**2)
    zPos[i] += zVel[i]*dt + 0.5*zForce0[i]*(dt**2)

  for i in range(nPart):
    xForce[i] = force(i,0)
    yForce[i] = force(i,1)
    zForce[i] = force(i,2)
  
  #velocity verlet velocity formula
  for i in range(nPart):
    xVel[i] += 0.5*(xForce[i] + xForce0[i])*dt
    yVel[i] += 0.5*(yForce[i] + yForce0[i])*dt
    zVel[i] += 0.5*(zForce[i] + zForce0[i])*dt
  
  #keep particles in PBC cell
  for i in range(nPart):
    xPos[i] -= lBox*(int(round(xPos[i]/lBox)))
    yPos[i] -= lBox*(int(round(yPos[i]/lBox)))
    zPos[i] -= lBox*(int(round(zPos[i]/lBox)))

#***************************************************************************************************************************************

def force(iPart, iComp):
  """
  Computes the force. We derived the formula for the force from the formula for the LJV. 
  """

  compForce = 0.0
  rSep = np.zeros(3)
  for jPart in range(nPart):
    if iPart != jPart:
      dr = partdist(xPos, yPos, zPos, iPart, jPart)
      if dr < rCut:
        magForce = epsilon*(48.0*(sigma**12/dr**14) - 24.0*(sigma**6/dr**8))
        compForce += magForce*rSep[iComp]
  return compForce

#***************************************************************************************************************************************
def measure_instant():
  """
  Stores the intantaneous positions and velocities.
  """
  pos=open("positions.dat",'a')
  vel=open("velocities.dat",'a')

  for i in range(nPart):
    pos.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(xPos[i],yPos[i],zPos[i]))
    vel.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(xVel[i],yVel[i],zVel[i]))

  pos.close()
  vel.close()

#***************************************************************************************************************************************

def measure():
  """
  Measures the static observables. We measure the numbers: the potential energy, the kinetic energy, and the virial. We measure the
  function(s) of distance r: the radial distribution function g(r).
  """  
  
  v = 0.0
  w = 0.0
  t = 0.0
  global vShift

  #reset the histogram of g(r)
  for i in range(igofr, igofr+nBin):
    walker[i]=0.0

  for i in range(nPart-1):
    for j in range(i+1,nPart):
      dr = partdist(xPos, yPos, zPos, i, j)

      #g(r)
      ibin = igofr + int(dr/sizeBin)
      if ibin<(igofr + nBin):
        walker[ibin] += 2.0

      if dr<rCut:
        vij = 4.0*epsilon*((sigma/dr)**12 - (sigma/dr)**6) - vCut
        wij = 1.0*((sigma/dr)**12) - 0.5*((sigma/dr)**6)

        #Potential energy
        v += vij
        #Virial 
        w += wij
        #Shifted potential
        vShift += vCut

  #Kinetic energy
  for i in range(nPart):
    t += 0.5*(xVel[i]**2 + yVel[i]**2 + zVel[i]**2)

  walker[iv] = v
  walker[iw] = 48.0*w/3.0
  walker[it] = t
  walker[ie] = walker[iv] + walker[it]

#***************************************************************************************************************************************
def measure_dyn(m):
  """
  Measures the dynamical observables. We measure the function(s) of time: the velocity autocorrelation function (VACF). We measure the
  function(s) of time and momentum: the spatial Fourier transform of the density autocorrelation function (F_r[DACF]). 
  
  To calculate the VACF we have at least two strategies available. Within each step t, of which there are nStep many, we keep the
  system moving over a time nTdelay by including an additional nTdelay time steps and compute the
  dynamical variables. We assume the system is ergodic, which implies we can replace ensemble averages with time averages. One
  strategy we could do is to perform a time average for every individual particle to get the VACF, for example, for that particular
  particle. To get even better results we could then average that over the number of particles. The time average is defined with an
  integral, but we would approximate it with a sum. However, we could make another approximation and just perform the average over the
  number of particles, for all time between 0 and nTdelay*dt. This is justified because the particle number average behaviour of some
  quantity that depends on the motion of the particles serves as a stand in for the behavior of this quantity for a "typical"
  particle. This is the strategy we use in the code. Its advantage is that it allows us to calculate the VVAC while the simulation is
  running. In the previous strategy we would have to complete the simulation and perform the calculation on the data we had generated
  after the simulation was complete.

  To calculate the DACF, we employ the following strategy. First note that the DACF is a function of time and distance. We want to use the
  DACF to get the dynamical structure factor (S(q,w)), which is a function of momentum and frequency. So instead of calculating the DACF
  directly, we calculate its spacial Fourier transform, F_r[DACF].

  To be continued...
  
  """  

  #velocity autocorrelation
  im = icvel + m #run through all the indices in walker reserved for VACF, starting with icvel, with nTdelay many to run through
  if m==0: #get the initial velocity we will which we correlate with later velocities
    for i in range(nPart):
      xVel0[i] = xVel[i]
      yVel0[i] = yVel[i]
      zVel0[i] = zVel[i]

  walker[im]=0.0 #initialize the section of walker reserved for the VACF
  for i in range(nPart): #average over particle number
    walker[im] += xVel[i]*xVel0[i] + yVel[i]*yVel0[i] + zVel[i]*zVel0[i]
  walker[im] = walker[im]/nPart

  #spacial Fourier transform of density autocorrelation
  if m==0:
    for i in range(nMomenta): 
      qRho0[i] = complex(0,0)
      for j in range(nPart):
        qRho0[i] += cmath.exp(complex(0,qMomenta[i]*xPos[j])) #negative in exp is gone because it cancels with negative q in definition of qRho(t)

  for i in range(nMomenta):
    qRho[i] = complex(0,0)
    for j in range(nPart):
      qRho[i] += cmath.exp(complex(0,-qMomenta[i]*xPos[j]))

  for i in range(nMomenta):
    im = icrho + i + m*nMomenta
    walker[im] = 0.0

  for i in range(nMomenta):
    im = icrho + i + m*nMomenta
    corr = qRho[i]*qRho0[i]
    walker[im] += corr.real
    walker[im] = walker[im]/nPart

#***************************************************************************************************************************************
def average(iwhat, whichStep):
  """
  Computes averages. Each block is an approximately independent measurement, if nStep is large enough, of some observable that is a
  function of the positions and velocities of the particles, like the average kinetic energy, or the virial, for example. Each run of
  the simulation, is an experiment which is a set of measurements. We print the values averaged over a single block to a file, then the
  file "statistics.py" handles the calculation of standard errors.
  
  ...
  
  remember blockNorm is the same as nStep
  """

  wd = 12
  global blockNorm, blockAverage, experimentAverage, walker
  
  #Reset block averages
  if iwhat == 1:
    blockAverage = np.zeros(nProp)
    experimentAverage = np.zeros(nProp)
    blockNorm = 0.0

  #Update block averages
  elif iwhat == 2:
    print("Step number: ", whichStep)
    for i in range(nProp):
      blockAverage[i] = blockAverage[i] + walker[i]
    blockNorm = blockNorm + 1.0

  #Print results for current block
  elif iwhat == 3:
    print("Block number: ", iblk)

    Epot=open("epotential.dat",'a')
    Pres=open("pressure.dat",'a')
    Ekin=open("ekinetic.dat",'a')
    Temp=open("temperature.dat",'a')
    Etot=open("etotal.dat",'a')
    Gofr=open("gofr.dat",'a')
    Vacf=open("vacf.dat",'a')
    Ddcf=open("ddcf.dat",'a')

    #Average potential energy per particle
    Epot.write("{:f}\t\t\t{:f}\n".format(blockAverage[iv]/blockNorm/nPart,blockNorm))
    #Average pressure per particle
    Pres.write("{:f}\t\t\t{:f}\n".format(rho*(2.0/3.0)*blockAverage[it]/blockNorm/nPart+((blockAverage[iw]/blockNorm)/Vol),blockNorm))
    #Average kinetic energy per particle
    Ekin.write("{:f}\t\t\t{:f}\n".format(blockAverage[it]/blockNorm/nPart,blockNorm))
    #Average temperature per particle
    Temp.write("{:f}\t\t\t{:f}\n".format((2.0/3.0)*blockAverage[it]/blockNorm/nPart,blockNorm))
    #Average total energy per particle
    Etot.write("{:f}\t\t\t{:f}\n".format(blockAverage[ie]/blockNorm/nPart,blockNorm))

    #g(r)
    for k in range(igofr,igofr+nBin):
      sd = 4.0*np.pi/3.0
      kk = k - igofr
      r = kk * sizeBin
      gdir = blockAverage[k]/blockNorm
      gdir *= 1.0/(sd*((r + sizeBin)**3 - (r**3))*rho*nPart)
      Gofr.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(gdir,blockNorm,r))

    for l in range(icvel,icvel+nTdelay):
      t = (l-icvel)*dt
      Vacf.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(blockAverage[l]/blockNorm,blockNorm,t))

    for j in range(icrho,icrho+nTdelay*nMomenta):
      t = int((j-icrho)/nMomenta)*dt
      q = (j - icrho)%nMomenta
      Ddcf.write("{:f}\t\t\t{:f}\t\t\t{:f}\t\t\t{:f}\n".format(blockAverage[j]/blockNorm,blockNorm,t,qMomenta[q]))

    Epot.close()
    Pres.close()
    Ekin.close()
    Temp.close()
    Etot.close()
    Gofr.close()
    Vacf.close()
    Ddcf.close()

    print("----------------------------")

#***************************************************************************************************************************************
#gets final configuration of system
def config_final():
  """
  Stores the final particle conditions. Adds tail corrections.
  """
  
  global vShift
  print("Print final configuration to file config.final \n\n")

  conf=open("config_final.dat",'a')
  for i in range(nPart):
    conf.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(xPos[i]/lBox,yPos[i]/lBox,zPos[i]/lBox))
  conf.close()

  vShift /= float(nStep*nBlock*nPart)
  
  print("Information for recovering the full LJ potential properties \n")
  print("Add to the potential and to the total energy the correction term ", vShift + vTail)
  print("Add to the pressure the correction term ", rho*pTail)
 
#***************************************************************************************************************************************
#main
#interface()
initialize()
for iblk in range(nBlock):
  average(1,0)
  for t in range(nStep):
    for m in range(nTdelay):
      measure_dyn(m)
      if (instFlag == 1):
        measure_instant()
      move()
    measure()
    average(2,t)
  average(3,0)
config_final()
#***************************************************************************************************************************************
