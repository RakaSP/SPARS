from .fcfs_auto_switch_on import FCFSAuto
from bisect import bisect_left


class EASYAuto(FCFSAuto):
    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        super().FCFSAuto()

        self.backfill()

        if self.timeout is not None:
            super().timeout_policy()

        return self.events

    def find_node_combination(self, p_start_t, compute_demand, nodes, next_releases, x):
        """
        Returns a tuple (best_combo, best_finish_time)
        - best_combo: list[dict node] or None
        - best_finish_time: float (0 if None)

        Idea:
        For a cutoff r (candidate "max activation delay"), a combo of size x is feasible iff
            min_speed >= compute_demand / (p_start_t - r)
        using only nodes with release_time <= r.
        We scan r across unique release times (ascending), keep eligible nodes in a list
        sorted by speed, and check feasibility with a bisect. This avoids combinations().
        """
        n = len(nodes)
        if x > n or x <= 0:
            return None

        # --- Precompute release times and filter impossible nodes early
        release_by_id = {e['id']: e['release_time'] for e in next_releases}
        cand = []
        for nd in nodes:
            s = float(nd.get('compute_speed', 0) or 0.0)
            if s <= 0:
                continue
            rid = nd['id']
            r = float(release_by_id.get(rid, 0.0))
            # If r >= p_start_t, this node can never finish a positive-demand job before p_start_t
            if r >= p_start_t:
                continue
            cand.append((r, s, nd))  # (release_time, speed, node_ref)

        if len(cand) < x:
            return None

        # Sort candidates by release_time ascending
        cand.sort(key=lambda t: t[0])
        unique_r = sorted(set(r for r, _, _ in cand))

        # We'll maintain a list of eligible nodes (those with release_time <= r), sorted by speed asc
        eligible = []              # list of (speed, node_ref, release_time)
        eligible_speeds = []       # parallel list of speeds for bisect
        idx_cand = 0
        best_finish_time = 0.0
        best_combo = None

        for r in unique_r:
            # Add all nodes with release_time == r into eligible
            while idx_cand < len(cand) and cand[idx_cand][0] <= r:
                ri, si, ndi = cand[idx_cand]
                # insert by speed (ascending)
                pos = bisect_left(eligible_speeds, si)
                eligible_speeds.insert(pos, si)
                eligible.insert(pos, (si, ndi, ri))
                idx_cand += 1

            # Compute required minimum speed at this cutoff
            remain = p_start_t - r
            if remain <= 0:
                continue  # impossible at this r

            min_speed_req = compute_demand / remain

            # Count how many eligible nodes have speed >= min_speed_req
            pos = bisect_left(eligible_speeds, min_speed_req)
            avail = len(eligible_speeds) - pos
            if avail < x:
                continue  # not feasible at this r

            # Choose the x *slowest* nodes that still meet the threshold (tight finish time)
            # [(speed, node, release_time), ...]
            chosen = eligible[pos: pos + x]
            speeds = [t[0] for t in chosen]
            nodes_sel = [t[1] for t in chosen]
            rel_times = [t[2] for t in chosen]

            s_min = min(speeds)                     # == eligible_speeds[pos]
            # may be <= r; use the actual max of chosen
            r_max = max(rel_times)
            finish_time = r_max + (compute_demand / s_min)

            # Keep the latest finish time that still fits
            if finish_time <= p_start_t and finish_time > best_finish_time:
                best_finish_time = finish_time
                best_combo = nodes_sel

        return best_combo

    def backfill(self):

        waiting_queue = [
            job for job in self.waiting_queue if job['job_id'] not in self.scheduled]
        if len(waiting_queue) > 2:
            p_job = waiting_queue[0]

            backfilling_queue = waiting_queue[1:]

            reserved_nodes = self.allocated
            reserved_node_ids = [node['id'] for node in reserved_nodes]
            not_reserved_nodes = [
                node for node in self.state if node['id'] not in reserved_node_ids]

            next_releases = self.resources_agenda

            next_releases = sorted(
                next_releases,
                key=lambda x: (x['release_time'], x['id'])
            )

            if len(next_releases) < p_job['res']:
                return

            last_host = next_releases[p_job['res'] - 1]
            p_start_t = last_host['release_time']

            candidates = [r['id']
                          for r in next_releases if r['release_time'] <= p_start_t]
            head_job_reservation_ids = candidates[-p_job['res']:]

            not_reserved_nodes = [
                r for r in not_reserved_nodes if r['id'] not in head_job_reservation_ids]

            for job in backfilling_queue:
                if job['job_id'] == 34061 and self.current_time == 1817:
                    print('x')
                allocated_ids = [node['id'] for node in self.allocated]
                not_computing_resource_ids = [
                    node['id'] for node in self.state if node['job_id'] is None and node['id'] not in allocated_ids]
                not_reserved_ids = [
                    h for h in not_computing_resource_ids if h not in head_job_reservation_ids]

                available_resources_not_reserved = [
                    node for node in self.available if node['id'] in not_reserved_ids]
                inactive_resources_not_reserved = [
                    node for node in self.inactive if node['id'] in not_reserved_ids]

                if job['res'] <= len(not_reserved_ids):
                    if job['res'] <= len(available_resources_not_reserved):
                        allocated_nodes = available_resources_not_reserved[:job['res']]

                        allocated_ids = [node['id']
                                         for node in allocated_nodes]

                        event = {
                            'job_id': job['job_id'],
                            'subtime': job['subtime'],
                            'runtime': job['runtime'],
                            'reqtime': job['reqtime'],
                            'res': job['res'],
                            'type': 'execution_start',
                            'nodes': allocated_ids
                        }
                        if self.timeout:
                            super().remove_from_timeout_list(allocated_ids)
                        self.available = [
                            node for node in self.available if node['id'] not in allocated_ids]
                        self.allocated.extend(allocated_nodes)
                        # if event['job_id'] == 25:
                        #     print('ea 109')
                        self.jobs_manager.add_job_to_scheduled_queue(
                            event['job_id'], allocated_ids, self.current_time)
                        super().push_event(self.current_time, event)
                        """should update releases agenda, do the same for others'
                        """
                    elif job['res'] <= len(available_resources_not_reserved) + len(inactive_resources_not_reserved):
                        count_avail = len(available_resources_not_reserved)
                        num_need_activation = job['res'] - count_avail
                        to_activate = inactive_resources_not_reserved[:num_need_activation]

                        allocated_nodes = available_resources_not_reserved
                        reserved_nodes = allocated_nodes + to_activate

                        self.allocated.extend(reserved_nodes)

                        self.available = [
                            node for node in self.available if node not in allocated_nodes]

                        self.inactive = self.inactive[num_need_activation:]

                        compute_demand = job['reqtime']

                        reserved_node_ids = [node['id']
                                             for node in reserved_nodes]

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
                        if self.timeout:
                            super().remove_from_timeout_list(reserved_node_ids)
                        # if event['job_id'] == 25:
                        #     print('ea 151')
                        to_activate_ids = [node['id'] for node in to_activate]
                        self.jobs_manager.add_job_to_scheduled_queue(
                            event['job_id'], reserved_node_ids, start_predict_time)
                        super().push_event(self.current_time, {
                            'type': 'switch_on', 'nodes': to_activate_ids})
                        super().push_event(self.current_time, {
                            'type': 'reserve', 'nodes': reserved_node_ids})
                        super().push_event(start_predict_time, event)
                else:
                    allocated_ids = [node['id']
                                     for node in self.allocated]
                    not_computing_resource_ids = [
                        node['id'] for node in self.state if node['job_id'] is None and node['id'] not in allocated_ids and node['state'] != 'switching_off' and node['state'] != 'switching_on']

                    compute_demand = job['reqtime']
                    free_nodes = [{'id': node['id'], 'compute_speed': node['compute_speed']}
                                  for node in self.state if node['id'] in not_computing_resource_ids and node['id'] not in self.reserved_ids]

                    combo = self.find_node_combination(
                        p_start_t, compute_demand, free_nodes, next_releases, job['res'])

                    if combo is None:
                        continue

                    # Use the combo directly
                    combo_ids = [n['id'] for n in combo]

                    # Which of the combo nodes are currently inactive?
                    inactive_id_set = {n['id'] for n in (
                        self.inactive or []) if isinstance(n, dict) and 'id' in n}
                    to_activate_ids = [
                        nid for nid in combo_ids if nid in inactive_id_set]

                    # Map id -> actual node objects from your pools so allocated contains real entries
                    id_to_node = {n['id']: n for n in (
                        self.available or []) if isinstance(n, dict) and 'id' in n}
                    id_to_node.update({n['id']: n for n in (self.inactive or [])
                                       if isinstance(n, dict) and 'id' in n})
                    reserved_nodes = [id_to_node[nid]
                                      for nid in combo_ids if nid in id_to_node]

                    # Book the reservation: move chosen nodes from pools to allocated
                    self.allocated.extend(reserved_nodes)
                    self.available = [
                        n for n in self.available if n['id'] not in combo_ids]
                    self.inactive = [
                        n for n in self.inactive if n['id'] not in combo_ids]

                    if self.timeout:
                        super().remove_from_timeout_list(combo_ids)

                    # Predicted start time = latest release among these nodes (fallback: now)
                    release_by_id = {ra['id']: ra['release_time']
                                     for ra in (self.resources_agenda or [])}
                    start_predict_time = max([release_by_id.get(nid, self.current_time)
                                              for nid in combo_ids] or [self.current_time])

                    event = {
                        'job_id': job['job_id'],
                        'subtime': job['subtime'],
                        'runtime': job['runtime'],
                        'reqtime': job['reqtime'],
                        'res': job['res'],
                        'type': 'execution_start',
                        'nodes': combo_ids,
                    }

                    # Queue/signal: switch on only inactive ones; always reserve; start at predicted time
                    self.jobs_manager.add_job_to_scheduled_queue(
                        event['job_id'], combo_ids, start_predict_time)
                    if to_activate_ids:
                        super().push_event(self.current_time, {
                            'type': 'switch_on', 'nodes': to_activate_ids})
                    super().push_event(self.current_time, {
                        'type': 'reserve', 'nodes': combo_ids})
                    super().push_event(start_predict_time, event)
