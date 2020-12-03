"""
Prototype Discrete Simulation Environment for RAM Modelling.

This model is extended from: https://simpy.readthedocs.io/en/latest/examples/machine_shop.html

Instead of being told a list of machines to feed in, it takes a DAG with nodes that indicate
the type of machine.

@author: Tyler Bikaun
"""

import random
import simpy
import numpy as np
import pandas as pd

# Helps reproduce results
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

from system_graph import SystemGraph


class MachineRepository:
    """ Accessor for all machinery information that can be selected within the simulation environment """
    def __init__(self):
        # PT_MEAN - Avg. processing time in minutes
        # PT_SIGMA - Sigma of processing time
        # MTTF - Mean time to failure in minutes
        # REPAIR_TIME - Time it takes to repair a machine in minutes
        self.MACHINE_CLASSES = {0: {'PT_MEAN': 10.0, 'PT_SIGMA': 2.0, 'MTTF': 300.0, 'REPAIR_TIME': 30.0},
                                1: {'PT_MEAN': 10.0, 'PT_SIGMA': 2.0, 'MTTF': 400.0, 'REPAIR_TIME': 30.0},
                                2: {'PT_MEAN': 10.0, 'PT_SIGMA': 2.0, 'MTTF': 50.0, 'REPAIR_TIME': 30.0}}

    def random_machine_class_choice(self):
        """
        Choose random machine classification
        """
        return random.choice(list(self.MACHINE_CLASSES.keys()))


class Machine(MachineRepository):
    """ 
    Logic: A machine will produces parts but may get broken every now and then. If it breaks, it requests a 'repairman' and resumes production once repaired.
    
    Current implementation includes:
        - Machines having distinct names, classifications and number of 'parts_made' thus far
    
    Arguments
    ---------
        env : TODO
            TODO
        name : str
            Name of machine object
        classification : str
            Classification of machine object
        repairman : TODO
            TODO
    Returns
    -------
    
    Notes
    -----
    
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
        """ Produce parts as long as the simulation is active. While making a part, the machine may break multiple times. If breakage occurs, a repairman is requested. 
        
        Arguments
        ---------
            repairman : TODO
                TODO
        Returns
        -------
        
        Notes
        -----
        
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
        """ Initiate machine breakage on time to failure parameter """
        while True:
            yield self.env.timeout(self.time_to_failure())  #
            if not self.broken:
                # Only break the machine if it is currently working.
                self.process.interrupt()

    def time_per_part(self):
        """ Return actual processing time for concrete part """
        return random.normalvariate(self.MACHINE_CLASSES[self.classification]['PT_MEAN'],
                                    self.MACHINE_CLASSES[self.classification]['PT_SIGMA'])
    
    def time_to_failure(self):
        """ Return time until next failure for a machine """
        break_mean = 1/self.MACHINE_CLASSES[self.classification]['MTTF']   # Param. for expovariate distribution
        return random.expovariate(break_mean)
    
    def observe(self):
        """ Captures machine state history """
        while True:
            self.stateObs.append([self.env.now, self.broken])
            yield self.env.timeout(1.0)  # Capture very 1 timestep
        

class DESEnv(MachineRepository):
    """ Discrete Event Simulation Environment (DESEnv)
    
    Arguments
    ---------
        system_graph : TODO
            Graph object of system configuration.
    """
    def __init__(self, system_graph):
        super().__init__()

        self.JOB_DURATION = 30.0    # Duration of other jobs in minutes
        self.NUM_MACHINES = 3       # Number of machines in the machine shop
        self.WEEKS = 4              # Simulation time in weeks
        self.SIM_TIME = self.WEEKS * 7 * 24 * 60    # Simulation time in minutes

        # List of machines and their classes for the simulation
        # Basic information without configuration atm. Will be updated to be
        # RDB call.
        # self.machine_dict = machine_dict
        self.system_graph = system_graph
        self.node_list = system_graph.get_node_details()    # used to get idx and label of machines in workshop

        # Execution
        self.create_env()
        self.run_env()
        self.analysis()
        self.availability()

    def other_jobs(self, env, repairman):
        """ Repairman's other (unimportant) jobs
        
        Arguments
        ---------
            env : TODO
                TODO
            repairman : TODO
                TODO
        Returns
        -------
        
        Notes
        -----
        
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
        """ Create an environment and start the set-up process """

        self.env = simpy.Environment()
        repairman = simpy.PreemptiveResource(self.env, capacity=1)

        # POPULATE MACHINES IN THE WORKSHOP USING SYSTEM DAG
        # Note: an agent will pass this information to simulation.
        self.machines = []
        for machine_no, machine_class in self.node_list.items():
            self.machines.append(Machine(self.env, f'Machine {machine_no}', machine_class, repairman))
        self.env.process(self.other_jobs(self.env, repairman))

    def run_env(self):
        self.env.run(until=self.SIM_TIME)

    def analysis(self):
        """ """
        
        print(f'Machine shop results after {self.WEEKS} weeks')
        for machine in self.machines:
            print(f'{machine.name} ({machine.classification}) made {machine.parts_made} parts')

        def system_stats(machines):
            # Average number of parts made
            return np.array([machine.parts_made for machine in self.machines]).mean()

        print(system_stats(self.machines))

    def availability(self):
        """ Calculates the system availability. Currently assumed to be in series configuration """

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
    # machine_dict = {0: 'C',
    #                 1: 'A',
    #                 2: 'B'}

    # A - adjacency matrix (node and edges)
    # an entire row of 1s indicates arrows away FROM the node,
    # an entire column of 1s indicates arrows TO the node
    A = np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0]])  # series configuration

    # F - feature list (node type)
    # row indicates node
    # col indicates node type
    F = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0]])

    # Generate system graph from adjacency matrix and feature list 
    sys_graph = SystemGraph(A, F)
    
    # Run discrete event simulation over system graph
    des_env = DESEnv(sys_graph)