
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
296
297
298
299
300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
324
325
326
327
328
329
330
331
332
333
334
335
336
337
338
339
340
341
342
343
344
345
346
347
348
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384
385
386
387
388
389
390
391
392
393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
439
440
441
442
443
444
445
446
447
448
449
450
451
452
453
454
455
456
457
458
459
460
461
462
463
464
465
466
467
468
469
470
471
472
473
474
475
476
477
478
479
480
481
482
483
484
485
486
487
488
489
490
491
492
493
494
495
496
497
498
499
500
501
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525
526
527
528
529
530
531
532
#immediate things to do:
#1. finish statistics
#2. add S(q,w)
#3. add final configuration
#4. add tail corrections
#5. add a simple user interface function
#6. clean up, make tidy
#7. add lots of comments
#8. go through a second time to check I understand everything

import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 
import os
import subprocess as sub
sub.call("rm -rf *.dat",shell=True)

#simulation parameters organized my use or type:
#**************************************************************
#The user will create different physical scenarios with these.
#macostate:
N=21 #number of particles
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
#user interface 
#makes code easier to use, run plotting and statistic from there
#def interface():
#**************************************************************

#setup initial conditions
def initialize():
  #Global Variabless
  global x,y,z,vx,vy,vz,vx0,vy0,vz0,pq,pq0

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
  x=np.zeros(size1)
  y=np.zeros(size1)
  z=np.zeros(size1)
  vx=np.zeros(size1)
  vy=np.zeros(size1)
  vz=np.zeros(size1)
  #v(t0) array for dynamical measurements
  vx0=np.zeros(size1)
  vy0=np.zeros(size1)
  vz0=np.zeros(size1)
  #pq(t) and pq0(t) arrays for dynamical measurements
  pq=np.empty(nq,dtype=np.complex_)
  pq0=np.empty(nq,dtype=np.complex_)


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
  for ix in range(nced):
    relx=ix*dimc
    for iy in range(nced):
      rely=iy*dimc
      for iz in range(nced):
        relz=iz*dimc
        for j in range(4):
          #perform the translation and place a particle there if
          #we have still not used up all the particles
          #changed N to N+N_hole, or else you will not start 
          #with full lattice to subtract off from
          #for no vacancies, N_hole==0
          if k<(size1):
            x[k] = relx + qfcc[j][0]*dimc
            y[k] = rely + qfcc[j][1]*dimc
            z[k] = relz + qfcc[j][2]*dimc
            #make PBC centered at origin
            x[k]+= - L*int(round(x[k]/L))
            y[k]+= - L*int(round(y[k]/L))
            z[k]+= - L*int(round(z[k]/L))
          k+=1

  #This part handles case when N_hole != 0.
  #We randomly delete N_hole many particles from full lattice
  #where N+N_hole is the closest "magic number" larger than N
  if hole_flag == 2:
    rnd_index=random.sample(range(size1), N_hole)
    x = np.delete(x, rnd_index)
    y = np.delete(y, rnd_index)
    z = np.delete(z, rnd_index)
    
  #initial velocities
  #calculate total potential energy from initial positions using LJV
  vtot=0.0
  for i in range(N-1):
    for j in range(i+1,N):
      vpair=LJV(x,y,z,i,j)
      vtot=vtot+vpair

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
    vx[i]=np.random.uniform()-0.5
    vy[i]=np.random.uniform()-0.5
    vz[i]=np.random.uniform()-0.5
    #net velocity components
    vnet[0]=vnet[0]+vx[i]
    vnet[1]=vnet[1]+vy[i]
    vnet[2]=vnet[2]+vz[i]
  
  #components of net velocity per particle
  vnet[0]=vnet[0]/N
  vnet[1]=vnet[1]/N
  vnet[2]=vnet[2]/N

  #subtract off net velocity of cloud from velocity of each particle
  for i in range(N):
    vx[i]=vx[i]-vnet[0]
    vy[i]=vy[i]-vnet[1]
    vz[i]=vz[i]-vnet[2]

  #calculate velocities using the thermo formula
  sumv2=0.0
  for i in range(N):
    sumv2+= vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]
  sumv2 /= N
  
  fs = (3*T/sumv2)**0.5
  #rescale velocities so that they satisfy the thermo formula
  for i in range(N):
    vx[i] *= fs
    vy[i] *= fs
    vz[i] *= fs

  print("(N,V,E) = ", N," ", vol," ", E)
  print("PBC cell side length: L = ", L)
  print("Total potential energy: epot = ", vtot)
  print("Total energy: E = ", E)
  print("Temperature: T = ", T)

#**************************************************************
  
#potential energy for a pair of particles function
#LJV(particles x positions, particles y positions, particles z positions, 1st particle index, 2nd particle index)
def LJV(q1, q2, q3, idi, idj):
  dx0=q1[idi]-q1[idj]
  dx0=dx0 - L*int(round(dx0/L))
  dy0=q2[idi]-q2[idj]
  dy0=dy0 - L*int(round(dy0/L))
  dz0=q3[idi]-q3[idj]
  dz0=dz0 - L*int(round(dz0/L))
  dr=dx0**2+dy0**2+dz0**2
  dr=dr**0.5
  v=4.0*epsilon*((sigma/dr)**12-(sigma/dr)**6)
  if dr>=rcut:
    v=0.0
  if dr<rcut:
    v=v-vcut
  return v

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
#dynamical measurements
#Within each step we keep the system moving over a time t_delay and 
#compute a dynamical variable, the VACF. Because the static observables
#are computed with ensemble averages (actually time averages that can 
#be taken as ensemble averages because the system is ergodic) we do 
#not need to do this and the procedure is less complicated. I think.
def measure_dyn(m):
  #velocity autocorrelation
  im=i_cv+m
  if m==0:
    for i in range(N):
      vx0[i]=vx[i]
      vy0[i]=vy[i]
      vz0[i]=vz[i]

  walker[im]=0.0
  for k in range(N):
    walker[im]+=vx[k]*vx0[k]+vy[k]*vy0[k]+vz[k]*vz0[k]
  walker[im]=walker[im]/N

  #dynamical density correlation
  if m==0:
    for i in range(nq):
      pq0[i]=complex(0,0)
      for j in range(N):
        #negative in exp is gone because it cancels with negative q in definitio of pq(t)
        pq0[i]+=cmath.exp(complex(0,qvec[i]*x[j]))


  for i in range(nq):
    pq[i]=complex(0,0)
    for j in range(N):
      pq[i]+=cmath.exp(complex(0,-qvec[i]*x[j]))


  print("Debug pq0 pq ",pq0[0], pq[0], qvec[0])

  for k in range(nq):
    im=i_pq+k+m*nq
    walker[im]=0.0

  for k in range(nq):
    im=i_pq+k+m*nq
    corr=pq[k]*pq0[k]
    walker[im]+=corr.real
    walker[im]=walker[im]/N
    if(k==0):
      print("Debug m ",m)
      print("Debug im ",im)
      print("Debug corr ",corr.real/N)
      print("Debug walker ",walker[im])
     

  #dynamical structure factor
  #make later



#**************************************************************

#computes averages
#need to change to averages and standard devs from different blocks
#recall the blk_norm is the same as nstep
#I do not know if the stdev is the same for every value of blk_av
#printed to file or whether it would be different
#I think it should be the same because each block is an experiments
#and the standard deviation comes from many experiments
def average(iwhat):
  wd=12
  global blk_norm,blk_av,exp_av,walker
  #Reset block averages
  if iwhat == 1:
    blk_av = np.zeros(n_props)
    exp_av = np.zeros(n_props)
    blk_norm = 0.0


  #Update block averages
  elif iwhat == 2:
    for i in range(n_props):
      blk_av[i] = blk_av[i] + walker[i]
    blk_norm = blk_norm + 1.0

  #Print results for current block
  elif iwhat == 3:
    print("Block number: ", iblk)

    Epot=open('epotential.dat','a') 
    Pres=open("pressure.dat",'a')
    Ekin=open("ekinetic.dat",'a')
    Temp=open("temperature.dat",'a')
    Etot=open("etotal.dat",'a')
    Gofr=open("gofr.dat",'a')
    Vacf=open("vacf.dat",'a')
    Ddcf=open("ddcf.dat",'a')


    #Average potential energy per particle
    Epot.write("{:f}\t\t\t{:f}\n".format(blk_av[iv]/blk_norm/N,blk_norm))
    #Average pressure per particle
    Pres.write("{:f}\t\t\t{:f}\n".format(rho*(2.0/3.0)*blk_av[it]/blk_norm/N+((blk_av[iw]/blk_norm)/vol),blk_norm))
    #Average kinetic energy per particle
    Ekin.write("{:f}\t\t\t{:f}\n".format(blk_av[it]/blk_norm/N,blk_norm))
    #Average temperature per particle
    Temp.write("{:f}\t\t\t{:f}\n".format((2.0/3.0)*blk_av[it]/blk_norm/N,blk_norm))
    #Average total energy per particle
    Etot.write("{:f}\t\t\t{:f}\n".format(blk_av[ie]/blk_norm/N,blk_norm))


    
    #g(r)
    for k in range(igofr,igofr+nbins):
      sd=4.0*np.pi/3.0
      kk = k - igofr
      r = kk * bin_size
      gdir = blk_av[k]/blk_norm
      gdir *= 1.0/(sd * ((r + bin_size)**3 - (r**3)) * rho * N)
      Gofr.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(gdir,blk_norm,r))
    
    for l in range(i_cv,i_cv+t_delay):
      t=(l-i_cv)*dt
      Vacf.write("{:f}\t\t\t{:f}\t\t\t{:f}\n".format(blk_av[l]/blk_norm,blk_norm,t))

    for j in range(i_pq,i_pq+t_delay*nq):
      t=int((j-i_pq)/nq)*dt
      q=(j-i_pq)%nq
      Ddcf.write("{:f}\t\t\t{:f}\t\t\t{:f}\t\t\t{:f}\n".format(blk_av[j]/blk_norm,blk_norm,t,qvec[q]))
    
    
    Epot.close()
    Pres.close()
    Ekin.close()
    Temp.close()
    Etot.close()
    Gofr.close()
    Vacf.close()
    Ddcf.close()

    print("----------------------------")

#**************************************************************
#gets final configuration of system
#def final_config():

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
