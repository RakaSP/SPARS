import logging
logger = logging.getLogger("runner")


class JobsManager:
    def __init__(self):
        self.waiting_queue = []
        self.scheduled_queue = []
        self.terminated_jobs = []
        self.finished_jobs = []
        self.active_jobs_id = []
        self.num_terminated_jobs = 0
        self.num_finished_jobs = 0

    def on_finish(self):
        for job in self.waiting_queue:
            self.remove_job_from_waiting_queue(job['job_id'], 'terminated')

        for job in self.scheduled_queue:
            self.remove_job_from_scheduled_queue(job['job_id'], 'terminated')

    def add_job_to_waiting_queue(self, job):
        self.waiting_queue.append(job)

    def remove_job_from_waiting_queue(self, job_id, type):
        for i, job in enumerate(self.waiting_queue):
            if job['job_id'] == job_id:
                if type == 'terminated':
                    self.terminated_jobs.append(job)
                    self.num_terminated_jobs += 1
                    logger.info(f'Job {job_id} is terminated')
                elif type == 'execution_start':
                    self.active_jobs_id.append(job_id)
                else:
                    raise ValueError(
                        f"Invalid removal type: '{type}' (expected 'terminated' or 'execution_start')")
                del self.waiting_queue[i]
                break

    def add_job_to_scheduled_queue(self, job_id, nodes):
        for job in self.waiting_queue:
            if job['job_id'] == job_id:
                job['nodes'] = nodes
                self.waiting_queue.remove(job)
                self.scheduled_queue.append(job)
                break

    def remove_job_from_scheduled_queue(self, job_id, type):
        for i, job in enumerate(self.scheduled_queue):
            if job['job_id'] == job_id:
                if type == 'fail':
                    for job in self.scheduled_queue:
                        if job['job_id'] == job_id:
                            del job['nodes']
                            del job['predicted_time']
                            self.add_job_to_waiting_queue(job)
                            break
                elif type == 'execution_start':
                    self.active_jobs_id.append(job_id)
                else:
                    raise ValueError(
                        f"Invalid removal type: '{type}' (expected 'terminated' or 'execution_start')")
                del self.scheduled_queue[i]
                break
