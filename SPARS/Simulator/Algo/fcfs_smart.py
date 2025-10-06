from math import inf
import math
from .BaseSmart import BaseSmart
import re
_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")


class FCFSSmart(BaseSmart):
    """
    First-Come-First-Served using only IDLE nodes.

    Node selection is energy-aware:
      Minimize ( sum(power) / min(compute_speed) ).
    Tie-breaks:
      1) Prefer switching_on nodes over sleeping nodes
      2) Shorter remaining idle-timeout first (closer to switch-off => pick sooner)
      3) Lower total power
      4) Lexicographically smaller node-id list
    Assumes each node has 'compute_speed' and 'power'.
    """

    def __init__(self, machines, jobs_manager, start_time, timeout=None):
        super().__init__(machines, jobs_manager, start_time, timeout)
        # Track scheduled jobs with their nodes, start_time, and finish_time
        self.selected_list = []

    # ---------- public ----------
    def schedule(self):
        if self.current_time == 106:
            print('here')
        super().prep_schedule()

        # First: Run FCFS scheduling
        fcfs_scheduled_jobs = self.FCFSSmart()

        # Then: Apply backfilling on remaining jobs
        self.backfill(fcfs_scheduled_jobs)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    def FCFSSmart(self):
        # This will now store tuples of (job, nodes, predicted_start_time, predicted_finish_time)

        # snapshot to avoid iterator issues

        # Track jobs scheduled by FCFS
        fcfs_scheduled_jobs = set()

        # First pass: try to schedule all jobs and collect their start times
        job_schedules = []  # Store (job, selected_nodes, start_time)

        for i, job in enumerate(self.waiting_queue[:]):
            if (self.current_time == 3670 or self.current_time == 3680) and job['job_id'] == 90:
                print('here')
            required = int(job["res"])

            # Get all currently scheduled nodes from job_schedules
            currently_scheduled_nodes = []
            for _, scheduled_nodes, _ in job_schedules:
                currently_scheduled_nodes.extend(scheduled_nodes)

            # 1) Prefer ACTIVE & IDLE
            candidates = list(self.idle)
            candidates = [
                candidate for candidate in candidates if candidate not in currently_scheduled_nodes]

            if len(candidates) >= required:
                selected = candidates[:required]
                # For immediate execution, start time is current time
                job_schedules.append((job, selected, self.current_time))
                fcfs_scheduled_jobs.add(job['job_id'])
                continue

            # 2) If not enough idle nodes, include ALL available nodes:
            candidates = (list(self.idle) + list(self.sleeping) +
                          list(self.computing) + list(self.switching_on))
            candidates = [
                candidate for candidate in candidates if candidate not in currently_scheduled_nodes]

            if len(candidates) >= required:
                result = self._select_nodes_energy_aware(required, candidates)
                if result is not None:
                    selected, start_time = result
                    job_schedules.append((job, selected, start_time))
                    fcfs_scheduled_jobs.add(job['job_id'])
                else:
                    # Can't schedule this job, so we break (FCFS)
                    break
            else:
                # Not enough nodes for this job
                break

        # Second pass: adjust start times to maintain FCFS order and prevent node switching
        adjusted_schedules = []
        max_start_time_so_far = 0

        for i, (job, selected, start_time) in enumerate(job_schedules):
            # Ensure jobs don't start earlier than previous jobs in FCFS order
            adjusted_start_time = max(start_time, max_start_time_so_far)
            adjusted_schedules.append((job, selected, adjusted_start_time))
            max_start_time_so_far = adjusted_start_time

        # Third pass: execute the schedules
        for job, selected, start_time in adjusted_schedules:
            # Calculate finish time for this job
            min_compute_speed = min(node['compute_speed'] for node in selected)
            finish_time = start_time + (job['reqtime'] / min_compute_speed)

            # Add to selected_list for timeout policy - now with finish_time
            self.selected_list.append((job, selected, start_time, finish_time))

            # Check if this is an immediate execution
            if start_time <= self.current_time:
                super().allocate(job, selected)
            else:
                # Schedule for future execution
                selected_ids = [n['id'] for n in selected]

                # Find sleeping nodes that need to be woken up
                sleeping_ids = {n['id'] for n in self.sleeping}
                switch_on_nodes = []
                for nid in selected_ids:
                    if nid in sleeping_ids:
                        switch_on_nodes.append(nid)

                if switch_on_nodes:
                    # Use the calculated start_time for switch_on events
                    self._schedule_switch_on_events(
                        job, selected, switch_on_nodes, start_time)

        return fcfs_scheduled_jobs

    def backfill(self, fcfs_scheduled_jobs):
        """EASY backfilling - backfill jobs that won't delay the first unscheduled job."""
        if len(self.waiting_queue) <= 1:
            return

        fcfs_selected = self.selected_list[:]

        # Get jobs not scheduled by FCFS
        unscheduled_jobs = [
            job for job in self.waiting_queue if job['job_id'] not in fcfs_scheduled_jobs]

        if not unscheduled_jobs:
            return

        # Get ALL reserved nodes from selected_list (not just head job)
        all_reserved_nodes = set()
        backfill_reserved_nodes = set()
        for job, nodes, start_time, finish_time in self.selected_list:
            all_reserved_nodes.update(node['id'] for node in nodes)

        # Step 1: Find the head job - prioritize future-scheduled jobs from selected_list
        head_job = None
        head_start_time = None
        head_finish_time = None
        head_nodes = None

        # Look for the latest job in selected_list that's scheduled in the future
        for job, nodes, start_time, finish_time in self.selected_list:
            if start_time > self.current_time:
                # This job is scheduled for future execution
                if head_start_time is None or start_time > head_start_time:
                    head_start_time = start_time
                    head_finish_time = finish_time
                    head_job = job
                    head_nodes = nodes

        # If no future-scheduled job found, use the first unscheduled job
        if head_job is None:
            head_job = unscheduled_jobs[0]
            # Calculate head job start time using enhanced next_releases
            enhanced_releases = self._get_enhanced_next_releases()

            if len(enhanced_releases) < head_job['res']:
                return  # Not enough nodes for head job

            # Use our node selection to find the best nodes for head job
            candidates = list(self.idle) + list(self.sleeping) + \
                list(self.computing) + list(self.switching_on)
            result = self._select_nodes_energy_aware(
                head_job['res'], candidates)

            if result is None:
                return  # Cannot schedule head job

            head_nodes, head_start_time = result
            # Calculate finish time for head job
            min_compute_speed = min(node['compute_speed']
                                    for node in head_nodes)
            head_finish_time = head_start_time + \
                (head_job['reqtime'] / min_compute_speed)

        # Try to backfill subsequent unscheduled jobs
        backfill_queue = unscheduled_jobs[1:
                                          ] if head_job in unscheduled_jobs else unscheduled_jobs

        for job in backfill_queue:
            if self.current_time == 3670 and job['job_id'] == 98:
                print('here')
            required = job['res']

            # Step 1: Try with idle nodes not reserved for ANY scheduled job
            candidates = list(self.idle)
            not_reserved = [
                candidate for candidate in candidates if candidate['id'] not in all_reserved_nodes]

            if len(not_reserved) >= required:
                # Enough non-reserved nodes available
                result = self._select_nodes_energy_aware(
                    required, not_reserved)
                if result is not None:
                    selected, start_time = result
                    # Backfill this job immediately
                    super().allocate(job, selected)
                    # UPDATE: Add to selected_list and update reserved nodes
                    min_compute_speed = min(
                        node['compute_speed'] for node in selected)
                    finish_time = start_time + \
                        (job['reqtime'] / min_compute_speed)
                    self.selected_list.append(
                        (job, selected, start_time, finish_time))
                    all_reserved_nodes.update(node['id'] for node in selected)
                    backfill_reserved_nodes.update(
                        node['id'] for node in selected)

                    continue

            # Step 2: Try with all idle nodes (including reserved ones) if job finishes before head job starts
            candidates = list(self.idle)
            candidates = [
                candidate for candidate in candidates if candidate['id'] not in backfill_reserved_nodes]
            if len(candidates) >= required:
                result = self._backfill_select_nodes_energy_aware(
                    job, required, candidates, fcfs_selected)
                if result is not None:
                    selected, start_time = result
                    super().allocate(job, selected)
                    # UPDATE: Add to selected_list and update reserved nodes
                    min_compute_speed = min(
                        node['compute_speed'] for node in selected)
                    finish_time = start_time + \
                        (job['reqtime'] / min_compute_speed)
                    self.selected_list.append(
                        (job, selected, start_time, finish_time))
                    all_reserved_nodes.update(node['id'] for node in selected)
                    backfill_reserved_nodes.update(
                        node['id'] for node in selected)
                    continue

            # Step 3: Include all nodes
            candidates = (list(self.idle) + list(self.sleeping) +
                          list(self.computing) + list(self.switching_on))

            # Filter out nodes reserved for ALL scheduled jobs
            candidates = [candidate for candidate in candidates
                          if candidate['id'] not in all_reserved_nodes]

            if len(candidates) >= required:
                if self.current_time == 1917 and job['job_id'] == 52:
                    print('here')
                result = self._select_nodes_energy_aware(
                    required, candidates)
                if result is not None:
                    selected, start_time = result

                    # Calculate finish time
                    min_compute_speed = min(
                        node['compute_speed'] for node in selected)
                    finish_time = start_time + \
                        (job['reqtime'] / min_compute_speed)

                    # UPDATE: Add to selected_list and update reserved nodes
                    self.selected_list.append(
                        (job, selected, start_time, finish_time))
                    all_reserved_nodes.update(node['id'] for node in selected)
                    backfill_reserved_nodes.update(
                        node['id'] for node in selected)

                    # Find sleeping nodes that need to be woken up
                    selected_ids = [n['id'] for n in selected]
                    sleeping_ids = {n['id'] for n in self.sleeping}
                    switch_on_nodes = []
                    for nid in selected_ids:
                        if nid in sleeping_ids:
                            switch_on_nodes.append(nid)

                    if switch_on_nodes:
                        # Use the calculated start_time for switch_on events
                        self._schedule_switch_on_events(
                            job, selected, switch_on_nodes, start_time)

                    continue

        # Step 4: Include ALL nodes and check if job can finish before head job starts
            candidates = (list(self.idle) + list(self.sleeping) +
                          list(self.computing) + list(self.switching_on))

            candidates = [
                candidate for candidate in candidates if candidate['id'] not in backfill_reserved_nodes]

            if len(candidates) >= required:
                result = self._backfill_select_nodes_energy_aware(
                    job, required, candidates, fcfs_selected)
                if result is not None:
                    selected, start_time = result

                    # Calculate finish time
                    min_compute_speed = min(
                        node['compute_speed'] for node in selected)
                    finish_time = start_time + \
                        (job['reqtime'] / min_compute_speed)

                    # UPDATE: Add to selected_list and update reserved nodes
                    self.selected_list.append(
                        (job, selected, start_time, finish_time))
                    all_reserved_nodes.update(node['id'] for node in selected)
                    backfill_reserved_nodes.update(
                        node['id'] for node in selected)

                    # Find sleeping nodes that need to be woken up
                    selected_ids = [n['id'] for n in selected]
                    sleeping_ids = {n['id'] for n in self.sleeping}
                    switch_on_nodes = []
                    for nid in selected_ids:
                        if nid in sleeping_ids:
                            switch_on_nodes.append(nid)

                    if switch_on_nodes:
                        # Use the calculated start_time for switch_on events
                        self._schedule_switch_on_events(
                            job, selected, switch_on_nodes, start_time)

                    continue

    def _get_enhanced_next_releases(self):
        """Create enhanced next_releases that includes FCFS-scheduled future jobs."""
        # Start with current next_releases
        enhanced_releases = self.next_releases.copy()

        # Add finish times from selected_list
        for job, nodes, start_time, finish_time in self.selected_list:
            if start_time > self.current_time:  # Future scheduled job
                # Update each node's release time
                for node in nodes:
                    found = False
                    for release in enhanced_releases:
                        if release['node_id'] == node['id']:
                            # Use the later release time
                            release['release_time'] = max(
                                release['release_time'], finish_time)
                            found = True
                            break

                    if not found:
                        enhanced_releases.append({
                            'node_id': node['id'],
                            'release_time': finish_time
                        })

        # Sort by release_time
        enhanced_releases.sort(key=lambda x: x['release_time'])
        return enhanced_releases

    def _backfill_select_nodes_energy_aware(self, job, required_nodes, candidates, fcfs_selected_list):
        if len(candidates) < required_nodes:
            return None

        # Precompute earliest start time for each node from FCFS jobs
        node_earliest_start = {}
        for fcfs_job, fcfs_nodes, start_time, finish_time in fcfs_selected_list:
            for node in fcfs_nodes:
                nid = node['id']
                if nid not in node_earliest_start or start_time < node_earliest_start[nid]:
                    node_earliest_start[nid] = start_time

        releases_by_id = super()._releases_by_id()
        machine_by_id = {m['id']: m for m in self.machines.machines}

        # Precompute per-node data
        node_data = {}
        for node in candidates:
            nid = node['id']
            node_release = releases_by_id[nid]
            machine = machine_by_id[nid]

            base_energy_waste = 0.0
            for q in node_release['queue']:
                if q['start_time'] < self.current_time:
                    duration = q['finish_time'] - self.current_time
                else:
                    duration = q['finish_time'] - q['start_time']

                if _COMPUTE_RE.fullmatch(str(q['phase'])):
                    continue

                e_rate = machine['states'][q['phase']]['power']
                if e_rate == 'from_dvfs':
                    dvfs_profiles = machine['dvfs_profiles']
                    dvfs_mode = node['dvfs_mode']
                    e_rate = dvfs_profiles[dvfs_mode]['power']

                base_energy_waste += e_rate * duration

            idle_power = machine['states']['active']['power']
            if idle_power == 'from_dvfs':
                dvfs_profiles = machine['dvfs_profiles']
                dvfs_mode = node['dvfs_mode']
                idle_power = dvfs_profiles[dvfs_mode]['power']

            # Calculate priorities
            state = node['state']
            if state == 'idle':
                state_priority = 0
            elif state == 'computing':
                state_priority = 1
            elif state == 'switching_on':
                state_priority = 2
            else:
                state_priority = 3

            if state == 'idle':
                timeout_priority = -self._remaining_idle_timeout(nid)
            else:
                timeout_priority = 0

            node_data[nid] = {
                'node': node,
                'base': float(base_energy_waste),
                'idle': float(idle_power),
                'release': float(node_release['release_time']),
                'state_priority': state_priority,
                'timeout_priority': timeout_priority,
            }

        # Generate all valid combinations
        valid_combos = []

        # Get all possible start times
        releases_sorted = sorted({data['release']
                                 for data in node_data.values()})

        for t in releases_sorted:
            # Get nodes available at time t
            available_nodes = []
            for nid, data in node_data.items():
                if data['release'] <= t:
                    # Calculate cost for this node at time t
                    if data['node']['state'] == 'switching_off' or data['node']['state'] == 'sleeping':
                        cost = data['base']
                    else:
                        cost = data['base'] + data['idle'] * \
                            (t - data['release'])

                    available_nodes.append({
                        'nid': nid,
                        'node': data['node'],
                        'cost': cost,
                        'state_priority': data['state_priority'],
                        'timeout_priority': data['timeout_priority'],
                    })

            if len(available_nodes) < required_nodes:
                continue

            # Sort available nodes by priority criteria
            available_nodes.sort(key=lambda x: (
                x['cost'],
                x['state_priority'],
                x['timeout_priority'],
                x['nid']
            ))

            # Generate combinations using the top nodes
            combo_nodes = [item['node']
                           for item in available_nodes[:required_nodes]]

            # Calculate finish time
            min_compute_speed = min(node['compute_speed']
                                    for node in combo_nodes)
            finish_time = t + (job['reqtime'] / min_compute_speed)

            # Check constraint
            max_allowed_finish = min(node_earliest_start.get(
                node['id'], float('inf')) for node in combo_nodes)

            if finish_time <= max_allowed_finish:
                total_cost = sum(item['cost']
                                 for item in available_nodes[:required_nodes])
                worst_state_priority = max(
                    item['state_priority'] for item in available_nodes[:required_nodes])
                worst_timeout_priority = min(
                    item['timeout_priority'] for item in available_nodes[:required_nodes])

                valid_combos.append({
                    'nodes': combo_nodes,
                    'start_time': t,
                    'total_cost': total_cost,
                    'worst_state_priority': worst_state_priority,
                    'worst_timeout_priority': worst_timeout_priority,
                    'node_ids': sorted(node['id'] for node in combo_nodes)
                })

        if not valid_combos:
            return None

        # Selection logic with multiple tie-breaking layers
        # Layer 1: Earliest start time
        min_start_time = min(combo['start_time'] for combo in valid_combos)
        tied_combos = [
            combo for combo in valid_combos if combo['start_time'] == min_start_time]

        if len(tied_combos) == 1:
            best_combo = tied_combos[0]
            return (best_combo['nodes'], best_combo['start_time'])

        # Layer 2: Least energy waste
        min_cost = min(combo['total_cost'] for combo in tied_combos)
        tied_combos = [
            combo for combo in tied_combos if combo['total_cost'] == min_cost]

        if len(tied_combos) == 1:
            best_combo = tied_combos[0]
            return (best_combo['nodes'], best_combo['start_time'])

        # Layer 3: Best timeout priority (most negative)
        best_timeout = min(combo['worst_timeout_priority']
                           for combo in tied_combos)
        tied_combos = [
            combo for combo in tied_combos if combo['worst_timeout_priority'] == best_timeout]

        if len(tied_combos) == 1:
            best_combo = tied_combos[0]
            return (best_combo['nodes'], best_combo['start_time'])

        # Layer 4: Best state priority (lowest number)
        best_state = min(combo['worst_state_priority']
                         for combo in tied_combos)
        tied_combos = [
            combo for combo in tied_combos if combo['worst_state_priority'] == best_state]

        if len(tied_combos) == 1:
            best_combo = tied_combos[0]
            return (best_combo['nodes'], best_combo['start_time'])

        # Final tie-break: lexicographically smallest node IDs
        best_combo = min(tied_combos, key=lambda x: x['node_ids'])
        return (best_combo['nodes'], best_combo['start_time'])

    def _schedule_switch_on_events(self, job, selected_nodes, switch_on_nodes, job_start_time):
        """
        Schedule switch_on events using call_me_later for future events
        and immediate switch_on for current time events.
        """
        immediate_switch_on = []
        future_switch_on_times = set()

        if self.current_time == 658:
            print('here')

        for node_id in switch_on_nodes:
            # Calculate when to start switching on this node
            switch_on_duration = super()._transition_time(node_id, 'switching_on', 'active')
            switch_on_start_time = job_start_time - switch_on_duration

            if switch_on_start_time <= self.current_time:
                # Immediate switch_on
                immediate_switch_on.append(node_id)
            else:
                # Future switch_on - schedule call_me_later
                future_switch_on_times.add(switch_on_start_time)

        # Handle immediate switch_on
        if immediate_switch_on and self.current_time == 658 and 15 in switch_on_nodes:
            print('here')
        if immediate_switch_on:
            def _filter_out(lst): return [
                n for n in lst if n['id'] not in immediate_switch_on]
            self.sleeping = _filter_out(self.sleeping)
            state_by_id = {n['id']: n for n in self.state}
            switch_on_nodes_list = []
            for node_id in immediate_switch_on:
                switch_on_nodes_list.append(state_by_id[node_id])
            self.switching_on.extend(switch_on_nodes_list)

            self.push_event(self.current_time, {
                'type': 'switch_on',
                'nodes': immediate_switch_on
            })

        # Handle future switch_on via call_me_later
        for switch_on_time in future_switch_on_times:
            self.push_event(switch_on_time, {
                'type': 'call_me_later'
            })

    # ---------- internals ----------

    def _remaining_idle_timeout(self, node_id: int) -> float:
        """
        Remaining time until this idle node would be switched off by timeout_policy.
        If not tracked, return a large number so it sorts to the end.
        """
        if self.timeout is None:
            return math.inf

        for entry in self.timeout_list:
            if entry["node_id"] == node_id:
                remaining = float(entry["time"] - self.current_time)
                return remaining

        return math.inf

    def _select_nodes_energy_aware(self, required_nodes: int, _candidates):
        if len(_candidates) < required_nodes:
            return None

        releases_by_id = super()._releases_by_id()

        # Precompute machine lookup
        machine_by_id = {m['id']: m for m in self.machines.machines}

        # Precompute per-node invariants: base, idle, release, and the node itself
        node_power_data = {}
        for node in _candidates:
            nid = node['id']
            node_release = releases_by_id[nid]
            machine = machine_by_id[nid]

            # Base energy waste from queued non-compute phases
            base_energy_waste = 0.0
            for q in node_release['queue']:
                # duration from current_time to finish if already started, else full duration
                if q['start_time'] < self.current_time:
                    duration = q['finish_time'] - self.current_time
                else:
                    duration = q['finish_time'] - q['start_time']

                # skip compute phase
                if _COMPUTE_RE.fullmatch(str(q['phase'])):
                    continue

                e_rate = machine['states'][q['phase']]['power']
                if e_rate == 'from_dvfs':
                    dvfs_profiles = machine['dvfs_profiles']
                    dvfs_mode = node['dvfs_mode']
                    e_rate = dvfs_profiles[dvfs_mode]['power']

                base_energy_waste += e_rate * duration

            # Idle power (active state, possibly DVFS)
            idle_power = machine['states']['active']['power']
            if idle_power == 'from_dvfs':
                dvfs_profiles = machine['dvfs_profiles']
                dvfs_mode = node['dvfs_mode']
                idle_power = dvfs_profiles[dvfs_mode]['power']

            node_power_data[nid] = {
                'base': float(base_energy_waste),
                'idle': float(idle_power),
                'release': float(node_release['release_time']),
                'node': node,
            }

        # Evaluate only distinct predicted start times t from release times
        releases_sorted = sorted({d['release']
                                  for d in node_power_data.values()})

        items = list(node_power_data.items())  # (nid, data)

        for t in releases_sorted:
            # list of (nid, cost_at_t, state_priority, timeout_priority)
            eligible = []

            for nid, dat in items:
                r = dat['release']
                if r <= t:
                    # KEEP ORIGINAL COST CALCULATION WITHOUT STATE PREFERENCE
                    if dat['node']['state'] == 'switching_off' or dat['node']['state'] == 'sleeping':
                        cost = dat['base']
                    else:
                        cost = dat['base'] + dat['idle'] * (t - r)

                    # State priority: idle > computing > switching_on > sleeping
                    state = dat['node']['state']
                    if state == 'idle':
                        state_priority = 0
                    elif state == 'computing':
                        state_priority = 1
                    elif state == 'switching_on':
                        state_priority = 2
                    else:  # sleeping
                        state_priority = 3

                    # Timeout priority: longer remaining idle-timeout first (further from switch-off)
                    if state == 'idle':
                        # Negative for reverse sort
                        timeout_priority = -self._remaining_idle_timeout(nid)
                    else:
                        timeout_priority = 0

                    eligible.append(
                        (nid, cost, state_priority, timeout_priority))

            if len(eligible) < required_nodes:
                continue

            # Pre-sort eligible by (cost, state_priority, timeout_priority, nid) for deterministic tie-breaking
            ranked = sorted((cost, state_priority, timeout_priority, nid)
                            for (nid, cost, state_priority, timeout_priority) in eligible)

            combo = []

            for cost, state_priority, timeout_priority, nid in ranked:
                combo.append(nid)
                if len(combo) == required_nodes:
                    break

            if len(combo) == required_nodes:
                # Convert node IDs back to node objects
                node_objects = [node_power_data[nid]['node'] for nid in combo]
                return (node_objects, t)
            else:
                return None
