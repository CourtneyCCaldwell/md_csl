import os
#delete old data files, always run this cell before next simulation run
os.remove("positions.dat")
os.remove("velocities.dat")
os.remove("epotential.dat") 
os.remove("pressure.dat")
os.remove("ekinetic.dat")
os.remove("temperature.dat")
os.remove("etotal.dat")
os.remove("gofr.dat")
os.remove("vacf.dat")
os.remove("statistics.dat")
