#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:38:15 2019

@author: bmoseley
"""

import shutil
import sys
import numpy as np
import multiprocessing, queue
import time
sys.path.insert(0, '../shared_modules/')
import io_utils
from helper import run_command




### TODO:
#
# Consider writing my own fortran code e.g using f2py
# to keep everything in memory
# it appears fortran is much quicker than numpy for seismiccpml

#### PARAMETERS

VEL_RUN = "layers"
SIM_RUN = "layers_2ms"

#VEL_RUN = "fault"
#SIM_RUN = "fault_2ms_r"

if 'linux' in sys.platform.lower(): ROOT_DIR = "/data/greypartridge/not-backed-up/aims/aims17/bmoseley/DPhil/Mini_Projects/DIP/forward_seisnets_paper/generate_data/"
else: ROOT_DIR = ""

N_VELS = 50000
#N_VELS = 100000

SEED = 123

QC_FREQ = 5000 # when to output full wavefield or not (I/O heavy)
n_processes = 6

#QC_FREQ = 10000 # when to output full wavefield or not (I/O heavy)
#n_processes = 42


## For WAVENET
RANDOMSOURCE = False
N_REC = 11
DELTAREC = 50.0 # receiver spacing (m)

## For AUTOENCODER
#RANDOMSOURCE = True
#N_REC = 32
#DELTAREC = 15.0 # receiver spacing (m)


##
SOURCE_Yi = 14# source depth
##
NX = 128
NY = 128
DELTAX = 5. # grid spacing (m)
DELTAY = 5. # grid spacing (m)
NPOINTS_PML = 10 # number of PML points
NSTEPS = 512*4
DELTAT = 0.0005 # sample rate for FD modelling (s)
ds = 4# downsample factor (for pre-processing)
##


####





np.random.seed(SEED)


## Place sources
if RANDOMSOURCE:# 3 sources per velocity model
    source_is = np.random.randint(4+NPOINTS_PML, NX-NPOINTS_PML-4, (N_VELS, 3, 1))
    source_is = np.concatenate([source_is, SOURCE_Yi*np.ones((N_VELS, 3, 1), dtype=int)], axis=2)
else:# 1 source per velocity model
    source_is = np.concatenate([(NX//2)*np.ones((N_VELS, 1, 1), dtype=int), SOURCE_Yi*np.ones((N_VELS, 1, 1), dtype=int)], axis=2)


receiver_is = np.array([int(np.floor( (DELTAX*NX/2. -(N_REC-1)*DELTAREC/2. + i*DELTAREC) / DELTAX) ) for i in range(N_REC)])
receiver_is = np.concatenate([receiver_is.reshape((N_REC,1)), SOURCE_Yi*np.ones((N_REC, 1), dtype=int)], axis=1)

print(source_is)
print()
print(receiver_is)
print()


def generate_example(ivel):
    
    SIM_NUM = ivel
    VEL_FILE = VEL_DIR + "velocity_%.8i.txt"%(ivel) 

    if SIM_NUM % QC_FREQ == 0: OUTPUT_WAVEFIELD = 1
    else: OUTPUT_WAVEFIELD = 0# whether to output wavefield (I/O heavy!)
    
    # run a separate simulation for each source
    for isource,source_i in enumerate(source_is[SIM_NUM]):
    
        # create a temporary directory for simulation output (prevent I/O clash between processes)
        TEMP_OUT_SIM_DIR = OUT_SIM_DIR + str(SIM_NUM) + "/"
        io_utils.get_dir(TEMP_OUT_SIM_DIR)
    
        # create receiver file
        RECEIVER_FILE = TEMP_OUT_SIM_DIR + "receiver_ijs_%s_%i.txt"%(SIM_RUN,SIM_NUM)
        with open(RECEIVER_FILE,'w') as f:
            f.write("%i\n"%(N_REC))
            for rec_i in receiver_is: f.write("%i %i\n"%(rec_i[0]+1, rec_i[1]+1))# SEISMIC CPML uses indices starting at 1
     
        # create source file (single source)
        SOURCE_FILE = TEMP_OUT_SIM_DIR + "source_ijs_%s_%i.txt"%(SIM_RUN,SIM_NUM)
        with open(SOURCE_FILE,'w') as f:
            f.write("%i\n"%(1))
            f.write("%i %i\n"%(source_i[0]+1, source_i[1]+1))# SEISMIC CPML uses indices starting at 1
    
    
        # RUN FORWARD SIMULATION
        
        cmd = "./xben_seismic_CPML_2D_pressure_second_order " + \
            "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"%(
            NSTEPS,
            NX,
            NY,
            DELTAX,
            DELTAY,
            DELTAT,
            NPOINTS_PML,
            0,# SOURCE_X (m)
            0,# SOURCE_Z (m)
            SOURCE_FILE,
            VEL_FILE,
            TEMP_OUT_SIM_DIR,
            SIM_NUM,
            RECEIVER_FILE,
            OUTPUT_WAVEFIELD)
        
        return_code = run_command(cmd.split(" "),verbose=False) # run
        
        if return_code != 0:
            print("ERROR: Simulation %i, %i broke, check stderr"%(ivel, isource))
            # CLEAR INTERMEDIARY FILES (CAREFUL !)
            io_utils.remove_dir(TEMP_OUT_SIM_DIR)
            return False

        
        # IMPORT GATHER INTO NUMPY
        
        gather = np.zeros((N_REC, NSTEPS), dtype=np.float32)
        file = TEMP_OUT_SIM_DIR + "gather_%.8i.bin"%(SIM_NUM)
        # Read each binary gather file (MUCH QUICKER THAN READING TEXT FILES, beacause its directed)
        with open(file,'rb') as f:
            #Note SEISMIC_CPML double precision saved to 64 bit floats (!) we DOWNSAMPLE to 32 bit floats
            # count = number of items (==np.float64 values) to process)
            for irec in np.arange(N_REC): gather[irec,:] = np.fromfile(f, dtype=np.float64, count = NSTEPS).astype(np.float32)
        
        
        # PRE-PROCESSING
        gather_decimated = np.copy(gather)# important to copy
        gather_decimated = gather_decimated[:,::ds]# DOWNSAMPLE GATHER
        
        
        # SAVE
        np.save(OUT_SIM_DIR + "gather_%.8i_%.8i.npy"%(SIM_NUM,isource), gather_decimated)
        
        
        # IMPORT WAVEFIELDS INTO NUMPY (for QC)
        if OUTPUT_WAVEFIELD:
            wavefields = np.zeros((NSTEPS,NX,NY), dtype=np.float32)
            files = [TEMP_OUT_SIM_DIR + "wavefield_%.8i_%.8i.bin"%(SIM_NUM, i+1) for i in range(NSTEPS)]# SEISMIC CPML uses indices starting at 1
            for i in range(NSTEPS):
                # Read each binary wavefield file (MUCH QUICKER THAN READING TEXT FILES, beacause its directed)
                with open(files[i],'rb') as f:
                    #Note SEISMIC_CPML double precision saved to 64 bit floats (!) we DOWNSAMPLE to 32 bit floats
                    # count = number of items (==np.float64 values) to process)
                    for iz in np.arange(NY): wavefields[i,:,iz] = np.fromfile(f, dtype=np.float64, count = NX).astype(np.float32)
            
            np.save(OUT_SIM_DIR + "wavefields_%.8i_%.8i.npy"%(SIM_NUM,isource), wavefields)
            np.save(OUT_SIM_DIR + "gather_raw_%.8i_%.8i.npy"%(SIM_NUM,isource), gather)


        # CLEAR INTERMEDIARY FILES (CAREFUL !)
        io_utils.remove_dir(TEMP_OUT_SIM_DIR)
        
    return True


def worker_function(taskQ, resultQ):
    """Try to get a ivel from tastQ to run. If sucessful, run forward modelling and push result to resultQ.
    If taskQ is empty, terminate."""
    
    while True:
        try: ivel = taskQ.get(block=True, timeout=10)# try to get the next task, allow some time for process clash (ivel number)
        except queue.Empty: break# kill process if no more tasks left
        example = generate_example(ivel)
        resultQ.put(example)# push the example to the results queue


if __name__ == "__main__":

    # initiate
    VEL_DIR = ROOT_DIR + "velocity/" + VEL_RUN + "/"
    OUT_SIM_DIR = ROOT_DIR + "gather/" + SIM_RUN + "/"
    
    # clear output directory for all simulations
    io_utils.get_dir(OUT_SIM_DIR)

    #save copy of this script for future reference
    shutil.copyfile('generate_forward_simulations.py', OUT_SIM_DIR + 'generate_forward_simulations_%s.py'%(SIM_RUN))
    
    # save source, receiver positions
    np.save(OUT_SIM_DIR + "source_is.npy", source_is)
    np.save(OUT_SIM_DIR + "receiver_is.npy", receiver_is)
    
    # make queues
    taskQ = multiprocessing.Queue()
    resultQ = multiprocessing.Queue()
    
    # push simulations to queues
    for ivel in range(N_VELS): taskQ.put(ivel)
    workers = [multiprocessing.Process(target=worker_function, args=(taskQ, resultQ)) for _ in range(n_processes)] 
    for worker in workers: worker.start()
    
    # listen to results queue and write results to output
    count = 0
    start = start0 = time.time()
    while count != N_VELS:
        try:
            _ = resultQ.get(False)# try to get next result immediately
            count +=1
            print("example written (count: %i of %i)"%(count, N_VELS))
            if count % 100 == 0:
                rate = 100./(time.time()-start)
                total = (time.time()-start0)/60.
                print("%.2f examples /sec"%(rate))
                start = time.time()
                print("Worker status': %s"%([worker.exitcode for worker in workers]))
        except queue.Empty: pass
    
    print("%i examples written"%(count))
    print("Simulations complete")
    
