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


class MachineRepository:
    """
    Accessor for all machinery information that can be slected within the simulation environment.

    """
    def __init__(self):
        # PT_MEAN - Avg. processing time in minutes
        # PT_SIGMA - Sigma of processing time
        # MTTF - Mean time to failure in minutes
        # REPAIR_TIME - Time it takes to repair a machine in minutes
        self.MACHINE_CLASSES = {'A': {'PT_MEAN': 10.0, 'PT_SIGMA': 2.0, 'MTTF': 300.0, 'REPAIR_TIME': 30.0},
                                'B': {'PT_MEAN': 10.0, 'PT_SIGMA': 2.0, 'MTTF': 400.0, 'REPAIR_TIME': 30.0},
                                'C': {'PT_MEAN': 10.0, 'PT_SIGMA': 2.0, 'MTTF': 50.0, 'REPAIR_TIME': 30.0}}

    def random_machine_class_choice(self):
        """
        Choose random machine classification
        """
        return random.choice(list(self.MACHINE_CLASSES.keys()))


class Machine(MachineRepository):
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
                    done_in = 0     # set to 0 to exit the while loop
                
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
        return random.normalvariate(self.MACHINE_CLASSES[self.classification]['PT_MEAN'],
                                    self.MACHINE_CLASSES[self.classification]['PT_SIGMA'])
    
    def time_to_failure(self):
        """
        Return time until next failure for a machine.
        """
        break_mean = 1/self.MACHINE_CLASSES[self.classification]['MTTF']   # Param. for expovariate distribution
        return random.expovariate(break_mean)
    
    def observe(self):
        """
        Captures machine state history.
        """
        while True:
            self.stateObs.append([self.env.now, self.broken])
            yield self.env.timeout(1.0)  # Capture very 1 timestep
        

class DESEnv(MachineRepository):
    def __init__(self, machine_dict):
        super().__init__()

        self.JOB_DURATION = 30.0    # Duration of other jobs in minutes
        self.NUM_MACHINES = 3       # Number of machines in the machine shop
        self.WEEKS = 4              # Simulation time in weeks
        self.SIM_TIME = self.WEEKS * 7 * 24 * 60    # Simulation time in minutes

        # List of machines and their classes for the simulation
        # Basic information without configuration atm. Will be updated to be
        # RDB call.
        self.machine_dict = machine_dict

        # Execution
        self.create_env()
        self.run_env()
        self.analysis()
        self.availability()

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

        self.env = simpy.Environment()
        repairman = simpy.PreemptiveResource(self.env, capacity=1)

        # Populate machines in the workshop
        # via dictionary. Agent will pass this information to simulation.
        self.machines = []
        for machine_no, machine_class in self.machine_dict.items():
            self.machines.append(Machine(self.env, f'Machine {machine_no}', machine_class, repairman))
        self.env.process(self.other_jobs(self.env, repairman))

    def run_env(self):
        self.env.run(until=self.SIM_TIME)

    def analysis(self):
        print(f'Machine shop results after {self.WEEKS} weeks')
        for machine in self.machines:
            print(f'{machine.name} ({machine.classification}) made {machine.parts_made} parts')

        def system_stats(machines):
            # Average number of parts made
            return np.array([machine.parts_made for machine in self.machines]).mean()

        print(system_stats(self.machines))

    def availability(self):
        """
        Calculates the system availability. Currently assumed to be in series configuration.
        """

        df_store = pd.DataFrame()
        for machine in self.machines:
            time = [record[0] for record in machine.stateObs]
            state = [record[1] for record in machine.stateObs]
            df = pd.DataFrame(data={'machine': machine.name, 'class': machine.classification,
                                    'time': time, 'state': state})
            df_store = pd.concat([df_store, df])

        # Calculate system availability (linear assumption; sums all states together)
        df_system_state = df_store.groupby(['time'])['state'].sum()

        self.availability = ((df_system_state == 0).astype(int).sum()/self.SIM_TIME)

        print(f'System availability: {self.availability*100:0.2f}%')


if __name__ == '__main__':
    machine_dict = {0: 'C',
                    1: 'A',
                    2: 'B'}
    des_env = DESEnv(machine_dict)
