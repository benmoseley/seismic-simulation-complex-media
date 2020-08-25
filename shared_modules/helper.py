#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:54:31 2018

@author: bmoseley
"""


# This module defines various generic python helper functions.


import copy as python_copy
import subprocess
import time
import numpy as np
import signal



## Helper functions / classes

def dB(S, ref_power=1., inverse=False):
    'Convert the real amplitude array S to a dB power scale'
    #
    # L = 10log_10 (P1/P0)
    # P1 = 10^(L/10)*P0 #inverse
    
    ref_power = float(ref_power)

    if inverse:
        Pow = (10.**(S/10.))*ref_power
        return np.sqrt(Pow)
    else:
        Pow = S**2.
        L = 10.*np.log10(Pow/ref_power)
        return L
    

def run_command(command, verbose=False):
    """Submit a command to shell. Wait for result.
    If verbose, print stdout and stderr to screen or output file.
    Otherwise, redirect stdout to devnull, still printing stderr.
    """
    
    if verbose:
        # open outfile
        if type(verbose)==str: f = open(verbose, "w")
        
        # open process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

        # monitor process
        while True:
            
            return_code = process.poll()
            
            # read stdout in realtime
            
            #stdout,stderr = process.communicate()# this waits for process to terminate though..
            #
            # the below blocks on readline() to get the stdout stream
            #
            # to not block on readline could use threading, like this: https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
            #
            # To avoid deadlocks: careful to: add \n to output, flush stdout, use
            # readline() rather than read()
            
            #print("reading line..")
            process.stdout.flush()
            output = process.stdout.readline()# blocks until newline or EOF. Returns an empty string if EOF is hit immediately.
            if type(output)==bytes: output = output.decode()
            if output:# else end of file reached
                if type(verbose)==str:
                    f.write(output.strip()+"\n")
                    f.flush()
                else: print(output.strip())
            
            # break if stdout is fully read and process terminated
            if not output and return_code != None:
                break
            
        # print stderr
        for output in process.stderr.readlines():
            if type(output)==bytes: output = output.decode()
            if output:# else end of file reached
                if type(verbose)==str:
                    f.write(output.strip()+"\n")
                    f.flush()
                else: print(output.strip())
            
        # print result
        output = "Exit code: %s"%(return_code)
        if type(verbose)==str:
            f.write(output+"\n")
            f.flush()
            f.close()
        else: print(output)
        
        process.stdout.close()
        process.stderr.close()
        
        return return_code# return the return code
    
    else:
        # open process
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)# still print stderr
        
        # wait for it to complete
        return_code = process.wait()
        
        # print stderr
        for output in process.stderr.readlines():
            if type(output)==bytes: output = output.decode()
            if output: print(output.strip())
        
        process.stderr.close()
                
        return return_code# return the return code
    
    
class DictToObj:
    "Convert a dictionary into a python object"
    def __init__(self, copy=True, **kwargs):
        "Input dictionary by values DictToObj(**dict)"
        assert type(copy)==bool
        for key in kwargs.keys():
            if copy:
                item = python_copy.deepcopy(kwargs[key])
                key = python_copy.deepcopy(key)
            else:
                item = kwargs[key]
            self[key] = item
            
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, item):
        setattr(self, key, item)
        
    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+="%s: %s\n"%(k,self[k])
        return s
    
    
class Timeout:
    "Context manager to limit or periodically monitor the execution of a block of code"
    def __init__(self, seconds, timeout_fn=None, repeat=False, verbose=False):
        self.seconds = seconds
        self.timeout_fn = timeout_fn
        self.repeat = repeat
        self.verbose = verbose
        
    def _signal_handler(self, signum, frame):
        if self.verbose: print('Signal handler called with signal', signum)
        if self.timeout_fn != None:
            self.timeout_fn()
        else:
            raise TimeoutError# raise exception in the main thread
        if self.repeat: self.__enter__()# re-enter the block if repeat is true
        
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._signal_handler)# set handler for SIGALRM to be signal_handler
        signal.alarm(self.seconds)# send SIGALRM signal to main process in seconds

    def __exit__(self, *args):
        # __exit__ is always run (no matter what happens) just before any exceptions
        # which are both raised and not handled inside the with block
        signal.alarm(0)# disable alarm
        if self.verbose: print("Alarm disabled")
        
class Timer:    
    "Simple timer context manager"
    def __enter__(self, verbose=False):
        self.start = time.time()
        self.verbose = verbose
        return self# so we can access this using "with Timer as timer"
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.verbose: print("Time elapsed: %f s"%(self.interval))
        
if __name__ == "__main__":
    
    d = {"a":[1,2,3], "b":2}
    
    a = DictToObj(**d)
    b = DictToObj(copy=False, **d)
    b.fun = "fun"
    b["yo"] = "yo"
    
    print(a,b)
    d["a"][0]=10
    print(a,b)
    
    success = False
    try:
        with Timeout(1, verbose=True):
            time.sleep(2)# unhandled exception raised in this block
            success = True
    except TimeoutError:
        print("hi2")
    print(success)
    
    with Timeout(1, verbose=True):
        try:
            time.sleep(2)# handled exception raised in this block
        except TimeoutError:
            print("hi2")
            
    with Timer() as timer:
        time.sleep(2)
    print(timer.interval)
    
    with Timeout(1, timeout_fn=lambda: print("oh no!")):
        time.sleep(2)
        print("hi1")
    print("hi2")
    
    with Timeout(1, timeout_fn=lambda: print("oh no!"), repeat=True):
        time.sleep(5)
        print("hi1")
    print("hi2")