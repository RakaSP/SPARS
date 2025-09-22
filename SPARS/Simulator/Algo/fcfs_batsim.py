from SPARS.Simulator.Algo.BaseBatsim import BaseBatsim
from operator import itemgetter


class FCFSBatsim(BaseBatsim):
    def schedule(self):

        super().prep_schedule()
        self.FCFSBatsim()
        super().events_builder()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSBatsim(self):
        for job in self.waiting_queue[:]:
            if job['res'] <= len(self.available):
                allocated_nodes = self.available[:job['res']]
                super().allocate(job, allocated_nodes)
            else:
                break
