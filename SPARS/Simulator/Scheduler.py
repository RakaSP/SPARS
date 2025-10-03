from SPARS.Simulator.Algo.easy_auto_switch_on import EASYAuto
from SPARS.Simulator.Algo.easy_normal import EASYNormal
from SPARS.Simulator.Algo.fcfs_auto_switch_on import FCFSAuto
from SPARS.Simulator.Algo.fcfs_normal import FCFSNormal
from SPARS.Simulator.Algo.fcfs_batsim import FCFSBatsim
from SPARS.Simulator.Algo.easy_batsim import EASYBatsim
from SPARS.Simulator.Algo.fcfs_smart import FCFSSmart

ALGO_MAP = {
    'fcfs_auto': FCFSAuto,
    'fcfs_normal': FCFSNormal,
    'easy_normal': EASYNormal,
    'easy_auto': EASYAuto,
    'fcfs_batsim': FCFSBatsim,
    'easy_batsim': EASYBatsim,
    'fcfs_smart': FCFSSmart,
}


class Scheduler:
    def __init__(self, machines, jobs_manager, algorithm, start_time, timeout=None):
        AlgorithmClass = ALGO_MAP[algorithm.lower()]
        self.algorithm = AlgorithmClass(
            machines,
            jobs_manager,
            start_time,
            timeout
        )

    def schedule(self, current_time):
        self.algorithm.set_time(current_time)
        events = self.algorithm.schedule()
        return events
