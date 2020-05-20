"""
Prototype Discrete Simulation Environment for RAM Modelling.

This model is extended from: https://simpy.readthedocs.io/en/latest/examples/machine_shop.html

@author: Tyler Bikaun
"""

import random
import simpy
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)    # Helps reproduce results.

class Machine(DESEnv):
    """
    A machine produces parts and may get broken every now and then. 
    
    If it breaks, it requests a 'repairman' and continues the production after it's repaired.
    
    A machine has a 'name', 'classification', and a number of 'parts_made' thus far.
    """

    def __init__(self, env, name, classification, repairman):
        
        super().__init__()
        
        self.env = env
        self.name = name
        self.classification = classification
        self.parts_made = 0
        self.broken = False
        
        self.stateObs = []  # history of state observations
        
        # Execute
        self.process = env.process(self.working(repairman))
        env.process(self.break_machine())
        env.process(self.observe())
        
    def working(self, repairman):
        """
        Produce parts as long as the simulation is running.
        
        While making a part, the machine may break multiple times.
        
        Request a repairman when this happens.
        """
        
        while True:
            # Start making a new part
            done_in = self.time_per_part()
            while done_in:
                try:
                    # Working on the part
                    start = self.env.now
                    yield self.env.timeout(done_in)
                    done_in = 0 # set to 0 to exit the while loop
                
                except simpy.Interrupt:
                    # Machine has been broken and process interrupted
                    self.broken = True
                    done_in -= self.env.now - start     # How much time is left?
                    
                    # Request a repairman. THis will preempt it's "other_job"
                    with repairman.request(priority=1) as req:
                        yield req
                        yield self.env.timeout(self.MACHINE_CLASSES[self.classification]['REPAIR_TIME']) 

                    self.broken = False     # repaired
                
            # Part is completed
            self.parts_made += 1
        
    def break_machine(self):
        """
        Break machine every now and then.
        """
        while True:
            yield self.env.timeout(self.time_to_failure())  #
            if not self.broken:
                # Only break the machine if it is currently working.
                self.process.interrupt()

    def time_per_part(self):
        """
        Return actual processing time for concrete part.
        """
        return random.normalvariate(self.MACHINE_CLASSES[self.classification]['PT_MEAN'], self.MACHINE_CLASSES[self.classification]['PT_SIGMA'])
    
    def time_to_failure(self):
        """
        Return time until next failure for a machine.
        """
        BREAK_MEAN = 1/self.MACHINE_CLASSES[self.classification]['MTTF']   # Param. for expovariate distribution
        return random.expovariate(BREAK_MEAN)
    
    def observe(self):
        """
        Captures machine state history.
        """
        while True:
            self.stateObs.append([env.now, self.broken])
            yield env.timeout(1.0)  # Capture very 1 timestep
        

class DESEnv(Machine):
    def __init__(self):
        
        super().__init__()
        
        self.create_env()
        
        
        self.JOB_DURATION = 30.0    # Duration of other jobs in minutes
        self.NUM_MACHINES = 3       # Number of machines in the machine shop
        self.WEEKS = 4              # Simulation time in weeks
        self.SIM_TIME = self.WEEKS * 7 * 24 * 60    # Simulation time in minutes
        
        # PT_MEAN - Avg. processing time in minutes
        # PT_SIGMA - Sigma of processing time
        # MTTF - Mean time to failure in minutes
        # REPAIR_TIME - Time it takes to repair a machine in minutes
        self.MACHINE_CLASSES = {'A': {'PT_MEAN': 10.0,'PT_SIGMA': 2.0,'MTTF': 300.0, 'REPAIR_TIME': 30.0},
                                'B':{'PT_MEAN': 10.0,'PT_SIGMA': 2.0,'MTTF': 400.0, 'REPAIR_TIME': 30.0},
                                'C':{'PT_MEAN': 10.0,'PT_SIGMA': 2.0,'MTTF': 50.0, 'REPAIR_TIME': 30.0}}
        
        # Execution
        self.create_env()
        
    def other_jobs(self, env, repairman):
        """
        The repairman's other (unimportant) jobs.
        """
        
        while True:
            # Start a new job
            done_in = self.JOB_DURATION
            # Retry the job until it is done
            # It's priority is lower than that of the machine repairs.
            with repairman.request(priority=2) as req:
                yield req
                try:
                    start = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start
        
    def create_env(self):
        """
        Create an environment and start the setup process
        """
        
        env = simpy.Environment()
        repairman = simpy.PreemptiveResource(env, capacity=1)
        machines = []
        
    
    
    
    
    
    
if __name__ == '__main__':
    des_env = DESEnv()