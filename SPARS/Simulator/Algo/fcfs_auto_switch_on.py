from SPARS.Simulator.Algo.BaseAlgorithm import BaseAlgorithm


class FCFSAuto(BaseAlgorithm):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        self.FCFSAuto()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSAuto(self):
        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled]

        for job in waiting_queue:
            if job['res'] <= len(self.available) + len(self.inactive):
                if job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]
                    allocated_ids = [node['id'] for node in allocated_nodes]
                    self.available = self.available[job['res']:]
                    self.allocated.extend(allocated_nodes)

                    self.scheduled.append(job['job_id'])

                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'runtime': job['runtime'],
                        'reqtime': job['reqtime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': allocated_ids
                    }
                    if event['job_id'] == 34668:
                        input('fa 36')
                    if self.timeout:
                        super().remove_from_timeout_list(allocated_ids)

                    self.jobs_manager.add_job_to_scheduled_queue(
                        event['job_id'], allocated_ids, self.current_time)
                    super().push_event(self.current_time, event)
                else:
                    count_avail = len(self.available)
                    num_need_activation = job['res'] - count_avail
                    if len(self.inactive) < num_need_activation:
                        break

                    to_activate = self.inactive[:num_need_activation]

                    reserved_nodes = self.available + to_activate

                    allocated_nodes = reserved_nodes
                    self.allocated.extend(reserved_nodes)
                    self.available = []
                    self.inactive = self.inactive[num_need_activation:]

                    """should consider activation delay"""

                    reserved_node_ids = [node['id'] for node in reserved_nodes]

                    highest_release_time = max(
                        (ra["release_time"]
                         for ra in self.resources_agenda if ra["id"] in reserved_node_ids),
                        default=0
                    )

                    start_predict_time = highest_release_time

                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'runtime': job['runtime'],
                        'reqtime': job['reqtime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': reserved_node_ids
                    }
                    if event['job_id'] == 34668:
                        input('fa 78')

                    if self.timeout:
                        super().remove_from_timeout_list(reserved_node_ids)
                    to_activate_ids = [node['id'] for node in to_activate]

                    super().remove_from_timeout_list(reserved_node_ids)
                    self.jobs_manager.add_job_to_scheduled_queue(
                        event['job_id'], reserved_node_ids, start_predict_time)
                    super().push_event(self.current_time, {
                        'type': 'switch_on', 'nodes': to_activate_ids})
                    super().push_event(self.current_time, {
                        'type': 'reserve', 'nodes': reserved_node_ids})
                    super().push_event(start_predict_time, event)

            else:
                break
