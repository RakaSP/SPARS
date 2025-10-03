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

    # ---------- public ----------
    def schedule(self):
        if self.current_time == 2:
            print('here')
        super().prep_schedule()
        self.FCFSSmart()

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    def FCFSSmart(self):
        # This will now store tuples of (nodes, predicted_start_time)

        # snapshot to avoid iterator issues
        if self.current_time == 709:
            print('here')
        for i, job in enumerate(self.waiting_queue[:]):
            required = int(job["res"])

            # 1) Prefer ACTIVE & IDLE
            candidates = list(self.idle)
            # Get all currently selected nodes from self.selected_list
            currently_selected_nodes = []
            for nodes, _ in self.selected_list:
                currently_selected_nodes.extend(nodes)

            candidates = [
                candidate for candidate in candidates if candidate not in currently_selected_nodes]

            if len(candidates) >= required:
                selected = candidates[:required]
                # For immediate execution, start time is current time
                self.selected_list.append((selected, self.current_time))
                super().allocate(job, selected)  # Immediate job execution
                continue  # Move to next job

            # 2) If not enough idle nodes, include ALL available nodes:
            # idle, sleeping, computing, AND switching_on
            candidates = (list(self.idle) + list(self.sleeping) +
                          list(self.computing) + list(self.switching_on))
            candidates = [
                candidate for candidate in candidates if candidate not in currently_selected_nodes]

            if len(candidates) >= required:
                result = self._select_nodes_energy_aware(required, candidates)
                if result is not None:
                    selected, start_time = result
                    self.selected_list.append((selected, start_time))
                    selected_ids = [n['id'] for n in selected]

                    # Check if next job would start earlier
                    if i + 1 < len(self.waiting_queue):
                        next_job = self.waiting_queue[i + 1]
                        next_required = int(next_job["res"])
                        next_candidates = (list(self.idle) + list(self.sleeping) +
                                           list(self.computing) + list(self.switching_on))
                        # Get all currently selected nodes including the current selection
                        all_selected_nodes = []
                        for nodes, _ in self.selected_list:
                            all_selected_nodes.extend(nodes)
                        next_candidates = [
                            c for c in next_candidates if c not in all_selected_nodes]

                        if len(next_candidates) >= next_required:
                            next_result = self._select_nodes_energy_aware(
                                next_required, next_candidates)
                            if next_result is not None:
                                next_selected, next_start_time = next_result
                                if next_start_time < start_time:
                                    # Next job would start earlier, so break the loop
                                    # Remove the current job from self.selected_list since we're not scheduling it
                                    self.selected_list.pop()
                                    break

                    # Find sleeping nodes that need to be woken up
                    # Note: switching_on nodes are already in process, so we don't need to wake them up again
                    sleeping_ids = {n['id'] for n in self.sleeping}
                    switch_on_nodes = []
                    for nid in selected_ids:
                        if nid in sleeping_ids:
                            switch_on_nodes.append(nid)

                    if switch_on_nodes:
                        # Use the calculated start_time for switch_on events
                        self._schedule_switch_on_events(
                            job, selected, switch_on_nodes, start_time)
            else:
                break

    def _schedule_switch_on_events(self, job, selected_nodes, switch_on_nodes, job_start_time):
        """
        Schedule switch_on events using call_me_later for future events
        and immediate switch_on for current time events.
        """
        immediate_switch_on = []
        future_switch_on_times = set()

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

        # Create sets for quick state lookups
        switching_on_ids = {n['id'] for n in self.switching_on}
        sleeping_ids = {n['id'] for n in self.sleeping}

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

            # Add state preference: switching_on > sleeping > others
            state_preference = 0.0
            if nid in switching_on_ids:
                # Prefer switching_on (negative to reduce cost)
                state_preference = -0.001
            elif nid in sleeping_ids:
                state_preference = 0.001   # Penalize sleeping slightly

            node_power_data[nid] = {
                'base': float(base_energy_waste),
                'idle': float(idle_power),
                'release': float(node_release['release_time']),
                'node': node,
                'state_preference': state_preference,
            }

        # Evaluate only distinct predicted start times t from release times
        releases_sorted = sorted({d['release']
                                 for d in node_power_data.values()})

        items = list(node_power_data.items())  # (nid, data)

        # First layer: find the earliest start time with valid combinations
        best_start_time = None
        best_combos_at_earliest_time = []  # Store all combos at the earliest start time

        for t in releases_sorted:
            eligible = []  # list of (nid, cost_at_t)
            anchors = []   # subset of eligible with release == t

            for nid, dat in items:
                r = dat['release']
                if r <= t:
                    # Include state preference in the cost calculation
                    cost = dat['base'] + dat['idle'] * \
                        (t - r) + dat['state_preference']
                    eligible.append((nid, cost))
                    if r == t:
                        anchors.append((nid, cost))

            if len(eligible) < required_nodes or not anchors:
                continue

            # Pre-sort eligible by (cost, nid) for deterministic tie-breaking
            ranked = sorted((cost, nid) for (nid, cost) in eligible)

            # Find all valid combinations at this start time t
            combos_at_t = []

            for anchor_nid, anchor_cost in anchors:
                if required_nodes == 1:
                    combo = (node_power_data[anchor_nid]['node'],)
                    total_cost = anchor_cost
                    combos_at_t.append((combo, total_cost))
                    continue

                # Take (k-1) smallest excluding the anchor
                picked_ids = []
                sum_rest = 0.0
                for cost, nid in ranked:
                    if nid == anchor_nid:
                        continue
                    picked_ids.append(nid)
                    sum_rest += cost
                    if len(picked_ids) >= required_nodes - 1:
                        break

                if len(picked_ids) < required_nodes - 1:
                    continue

                total_cost = anchor_cost + sum_rest
                chosen_ids = [anchor_nid] + picked_ids[:required_nodes - 1]
                # Deterministic ordering (by id); no order_idx used anywhere
                chosen_ids.sort()
                combo = tuple(node_power_data[nid]['node']
                              for nid in chosen_ids)
                combos_at_t.append((combo, total_cost))

            if combos_at_t:
                # This is the earliest start time we found with valid combinations
                best_start_time = t
                best_combos_at_earliest_time = combos_at_t
                break  # We found the earliest start time, no need to check later times

        if not best_combos_at_earliest_time:
            return None

        # Second layer: among combos at the earliest start time, pick the one with minimum energy waste
        best_combo = None
        minimum_energy_waste = inf

        for combo, energy_waste in best_combos_at_earliest_time:
            if energy_waste < minimum_energy_waste:
                minimum_energy_waste = energy_waste
                best_combo = combo

        if best_combo is not None:
            return (best_combo, best_start_time)
        return None
