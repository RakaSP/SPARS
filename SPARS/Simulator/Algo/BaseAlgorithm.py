from operator import itemgetter


class BaseAlgorithm:
    """
    Per-node resource agenda:
      resource_agenda = [
        {
          'node_id': <int>,
          'queue': [
            {'phase': <str>, 'start_time': <float>, 'finish_time': <float>},
            ...
          ],
          'release_time': <float>,  # absolute sim time when node becomes ACTIVE & IDLE
        },
        ...
      ]

    States/Phases:
      Machine state (from self.state[*]['state']): 'active', 'sleeping', 'switching_on', 'switching_off'
      Head phases we track: 'switching_off', 'switching_on', 'sleep_to_active', 'compute(job=...)'

    Partitions we expose each scheduling tick (mutually exclusive):
      - self.reserved     : nodes whose id is listed in jobs_manager.scheduled_queue (not yet computing)
      - self.computing    : state=='active' and job_id is not None
      - self.idle         : state=='active' and job_id is None  (and not reserved)
      - self.sleeping     : state=='sleeping'                   (and not reserved)
      - self.switching_on : state=='switching_on'               (and not reserved)
      - self.switching_off: state=='switching_off'              (and not reserved)
    """

    # ---------------- Init ----------------
    def __init__(self, machines, jobs_manager, start_time, timeout=None):
        self.machines = machines
        self.jobs_manager = jobs_manager

        self.state = machines.nodes
        self.transitions = machines.machines_transition
        self.waiting_queue = jobs_manager.waiting_queue
        self.scheduled_queue = jobs_manager.scheduled_queue
        self.events = []
        self.current_time = float(start_time)
        self.timeout = timeout

        # New partitions
        self.reserved = []
        self.computing = []
        self.idle = []
        self.sleeping = []
        self.switching_on = []
        self.switching_off = []

        self.timeout_list = []
        self.next_timeout_at = None

        # resource agenda (rebuilt in prep_schedule)
        self.resource_agenda = [
            {'node_id': n['id'], 'queue': [],
                'release_time': self.current_time}
            for n in self.state
        ]

        # for call_me_later on node state changes
        self._old_state_sig = self._snapshot_state_sig()

    # ---------------- Events & time ----------------
    def push_event(self, timestamp, event):
        bucket = next(
            (x for x in self.events if x['timestamp'] == timestamp), None)
        if bucket:
            bucket['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})
            self.events.sort(key=itemgetter('timestamp'))

    def set_time(self, current_time):
        self.current_time = float(current_time)

    # ---------------- Helpers ----------------
    def _agenda_by_id(self):
        return {e['node_id']: e for e in self.resource_agenda}

    def _snapshot_state_sig(self):
        """Compact signature to detect state changes."""
        return {n['id']: (n.get('state'), n.get('job_id')) for n in self.state}

    @staticmethod
    def _sum_queue_abs(q):
        """Return the absolute finish time of last phase or 0 if empty (caller sets to now)."""
        return float(q[-1]['finish_time']) if q else 0.0

    def _remaining_time(self, total, started_at, now):
        """Remaining time in current phase; conservative if timestamps unknown."""
        if total is None:
            return 0.0
        if started_at is None:
            return max(0.0, float(total))
        return max(0.0, float(total) - max(0.0, now - float(started_at)))

    def _recompute_release_at(self, entry):
        entry['release_time'] = self._sum_queue_abs(
            entry['queue']) if entry['queue'] else self.current_time

    def _append_phase_abs(self, entry, phase, start_time, duration):
        st = float(start_time)
        ft = st + float(duration)
        entry['queue'].append(
            {'phase': phase, 'start_time': st, 'finish_time': ft})
        entry['release_time'] = ft

    def _cursor_from_queue(self, entry):
        """Where the next phase would start (end of last phase or now)."""
        return float(entry['queue'][-1]['finish_time']) if entry['queue'] else self.current_time

        # ---------------- Transitions lookup (from machines_transitions) ----------------
    def _ensure_transition_index(self):
        """
        Build once: { node_id: { (from_state, to_state): transition_time, ... }, ... }
        Expected external attribute: self.machines_transitions = [
            {"node_id": 1, "transitions": [{"from": "sleeping", "to": "active", "transition_time": 12.3}, ...]},
            ...
        ]
        """
        if hasattr(self, "_trans_index_built") and self._trans_index_built:
            return

        self._trans_index = {}
        source = getattr(self, "machines_transitions", None) or []
        for row in source:
            nid = row.get("node_id")
            tlist = row.get("transitions") or []
            by_pair = {}
            for t in tlist:
                frm = str(t.get("from"))
                to = str(t.get("to"))
                tt = float(t.get("transition_time", 0.0))
                by_pair[(frm, to)] = tt
            if nid is not None:
                self._trans_index[int(nid)] = by_pair

        self._trans_index_built = True

    def _transition_time(self, node_id: int, from_state: str, to_state: str, default: float = 0.0) -> float:
        """Return transition_time for (from_state -> to_state) for node_id; default if not found."""
        self._ensure_transition_index()
        by_pair = self._trans_index.get(int(node_id), {})
        return float(by_pair.get((from_state, to_state), default))

    # ---------------- Resource agenda builders ----------------
    def _prune_finished(self, entry):
        """Drop phases that ended at or before now."""
        now = self.current_time
        if entry['queue']:
            entry['queue'] = [seg for seg in entry['queue']
                              if float(seg['finish_time']) > now]

    def _ensure_head(self, entry, phase_name, start_at, duration):
        """
        Ensure the queue head matches the current physical phase.
        If start_at is None and a matching head exists, keep its timing.
        Otherwise, insert/replace with (now or start_at) + duration.
        """
        q = entry['queue']
        now = self.current_time
        if q and q[0]['phase'] == phase_name:
            if start_at is not None:
                q[0]['start_time'] = float(start_at)
                q[0]['finish_time'] = float(start_at) + float(duration)
        else:
            st = float(start_at) if start_at is not None else float(now)
            ft = st + float(duration)
            q.insert(0, {'phase': phase_name,
                     'start_time': st, 'finish_time': ft})

    def _rebuild_resource_agenda_global(self):
        ag_by_id = self._agenda_by_id()
        now = self.current_time

        for node in self.state:
            node_id = node['id']
            entry = ag_by_id.get(node_id)
            if entry is None:
                entry = {'node_id': node_id, 'queue': [], 'release_time': now}
                self.resource_agenda.append(entry)

            # prune finished
            self._prune_finished(entry)

            state = node.get('state')
            job_id = node.get('job_id')
            started_at = node.get('phase_started_at')  # optional timestamp

            # durations from machines_transitions
            switching_off_to_sleeping = self._transition_time(
                node_id, 'switching_off', 'sleeping', 0.0)
            switching_on_to_active = self._transition_time(
                node_id, 'switching_on',  'active',   0.0)

            if state == 'switching_off':
                st = float(started_at) if started_at is not None else None
                self._ensure_head(entry, 'switching_off', st,
                                  switching_off_to_sleeping)

            elif state == 'switching_on':
                st = float(started_at) if started_at is not None else None
                self._ensure_head(entry, 'switching_on', st,
                                  switching_on_to_active)

            elif state == 'sleeping':
                # steady; remove stray switching head phases
                if entry['queue'] and entry['queue'][0]['phase'] in ('switching_off', 'switching_on'):
                    entry['queue'].pop(0)

            elif state == 'active':
                if job_id is None:
                    # drop stray switching/sleep_to_active heads
                    while entry['queue'] and entry['queue'][0]['phase'] in ('switching_off', 'switching_on', 'sleep_to_active'):
                        entry['queue'].pop(0)
                else:
                    # if compute phase missing, allocator/events reconcile later
                    pass

            # recompute absolute release time
            self._recompute_release_at(entry)

    def _update_resource_agenda_partial(self, node_ids, extra_phase):
        """
        Append a phase (absolute) for selected nodes and refresh release_time.
        extra_phase: {'phase': str, 'duration': float}
        """
        ag_by_id = self._agenda_by_id()
        for nid in node_ids:
            entry = ag_by_id.get(nid)
            if entry is None:
                entry = {'node_id': nid, 'queue': [],
                         'release_time': self.current_time}
                self.resource_agenda.append(entry)
            cursor = self._cursor_from_queue(entry)
            self._append_phase_abs(
                entry, extra_phase['phase'], cursor, float(extra_phase['duration']))

    # ---------------- Events builder ----------------
    def events_builder(self):
        """
        - Emit execution_start for any job whose allocated nodes are ACTIVE & idle.
        - Power control: ONLY switch_on nodes that are both RESERVED and SLEEPING.
        - Does NOT append compute phases here (allocate() already handled that in your version).
        """
        node_by_id = {n['id']: n for n in self.state}
        reserved_node_ids = {
            nid for j in self.jobs_manager.scheduled_queue for nid in j['nodes']}

        # Try to start jobs whose allocated nodes are all active & idle
        for job in self.jobs_manager.scheduled_queue:
            node_ids = job['nodes']
            can_start = True
            for nid in node_ids:
                node = node_by_id.get(nid)
                if (node is None) or (node.get('state') != 'active') or (node.get('job_id') is not None):
                    can_start = False
                    break

            if can_start:
                # Just emit the execution_start event; do NOT touch resource_agenda here
                get = itemgetter('job_id', 'subtime',
                                 'runtime', 'reqtime', 'res')
                job_id, subtime, runtime, reqtime, res = get(job)
                self.push_event(self.current_time, {
                    'type': 'execution_start',
                    'job_id': job_id,
                    'subtime': subtime,
                    'runtime': runtime,
                    'reqtime': reqtime,
                    'res': res,
                    'nodes': node_ids,
                })

        # Power: auto-switch ON only nodes that are both reserved & sleeping
        sleeping_reserved = [nid for nid in reserved_node_ids
                             if node_by_id.get(nid, {}).get('state') == 'sleeping']
        if sleeping_reserved:
            self.push_event(self.current_time, {
                'type': 'switch_on',
                'nodes': sleeping_reserved
            })

    # ---------------- Allocation ----------------
    def allocate(self, job, allocated_nodes):
        """
        Add nodes to reserved (scheduled_queue), append wake (if needed) + compute phases with absolute times.
        This does NOT remove jobs from waiting_queue; the simulator will consume scheduled_queue.
        """
        if not allocated_nodes:
            return

        node_ids = [n['id'] for n in allocated_nodes]

        # Update partitions: remove from non-reserved buckets, add to reserved
        ids = set(node_ids)

        def _filter_out(lst):
            return [n for n in lst if n['id'] not in ids]
        self.idle = _filter_out(self.idle)
        self.sleeping = _filter_out(self.sleeping)
        self.switching_on = _filter_out(self.switching_on)
        self.switching_off = _filter_out(self.switching_off)
        self.reserved.extend(allocated_nodes)

        # Register with jobs_manager
        self.jobs_manager.add_job_to_scheduled_queue(job['job_id'], node_ids)

        # walltime via slowest node
        compute_speed = min(float(n['compute_speed']) for n in allocated_nodes)
        walltime = float(job['runtime']) / compute_speed

        ag_by_id = self._agenda_by_id()
        for n in allocated_nodes:
            entry = ag_by_id.get(n['id'])
            if entry is None:
                entry = {'node_id': n['id'], 'queue': [],
                         'release_time': self.current_time}
                self.resource_agenda.append(entry)

            cursor = self._cursor_from_queue(entry)

            # Need wake if sleeping/switching_off now, or if tail says switching_off
            tail_phase = entry['queue'][-1]['phase'] if entry['queue'] else None
            need_wake = (n.get('state') in ('sleeping', 'switching_off')) or (
                tail_phase in ('switching_off',))
            if need_wake:
                # use machines_transitions: sleeping -> active
                sleep_to_active = self._transition_time(
                    n['id'], 'sleeping', 'active', 0.0)
                if sleep_to_active > 0:
                    self._append_phase_abs(
                        entry, 'sleep_to_active', cursor, sleep_to_active)
                    cursor = entry['release_time']

            # Append compute phase
            self._append_phase_abs(
                entry, f'compute(job={job["job_id"]})', cursor, walltime)

    # ---------------- Timeout handling ----------------
    def remove_from_timeout_list(self, node_ids):
        ids = set(node_ids)
        self.timeout_list[:] = [
            ti for ti in self.timeout_list if ti.get('node_id') not in ids]

    def timeout_policy(self):
        if not self.timeout:
            return

        now = self.current_time
        expire_at = now + self.timeout

        state_by_id = {n['id']: n for n in self.state}
        reserved_ids = {n['id'] for n in self.reserved}
        timeout_ids = {t['node_id'] for t in self.timeout_list}

        # Remove timeouts for nodes that are now reserved
        if timeout_ids & reserved_ids:
            self.timeout_list = [
                t for t in self.timeout_list if t['node_id'] not in reserved_ids]
            timeout_ids -= reserved_ids

        # Add timeouts for idle active not-reserved nodes
        for node in self.state:
            idle = (node.get('state') == 'active' and node.get('job_id') is None)
            nid = node['id']
            if idle and nid not in reserved_ids and nid not in timeout_ids:
                self.timeout_list.append({'node_id': nid, 'time': expire_at})
                timeout_ids.add(nid)

        # Walk timeouts
        keep, switch_off, next_earliest = [], [], None
        for t in self.timeout_list:
            nid = t['node_id']
            node = state_by_id.get(nid)
            if node is None:
                continue
            if nid in reserved_ids:
                continue
            idle = (node.get('state') == 'active' and node.get('job_id') is None)
            if not idle:
                continue
            if now < t['time']:
                keep.append(t)
                next_earliest = t['time'] if next_earliest is None else min(
                    next_earliest, t['time'])
            else:
                switch_off.append(nid)

        self.timeout_list = keep

        if switch_off:
            self.push_event(now, {'type': 'switch_off', 'nodes': switch_off})

        if next_earliest is not None and getattr(self, 'next_timeout_at', None) != next_earliest:
            self.push_event(next_earliest, {'type': 'call_me_later'})
            self.next_timeout_at = next_earliest

    # ---------------- Partition & prep ----------------
    def _build_partitions(self):
        """Build mutually-exclusive node partitions."""
        self.reserved, self.computing = [], []
        self.idle, self.sleeping = [], []
        self.switching_on, self.switching_off = [], []

        scheduled_ids = {
            nid for j in self.jobs_manager.scheduled_queue for nid in j['nodes']}

        for node in self.state:
            nid = node['id']
            state = node.get('state')
            job_id = node.get('job_id')

            if job_id is not None and state == 'active':
                self.computing.append(node)
                continue

            if nid in scheduled_ids and job_id is None:
                # Node reserved for a future job (not yet started)
                self.reserved.append(node)
                continue

            if state == 'active' and job_id is None:
                self.idle.append(node)
            elif state == 'sleeping':
                self.sleeping.append(node)
            elif state == 'switching_on':
                self.switching_on.append(node)
            elif state == 'switching_off':
                self.switching_off.append(node)

    def prep_schedule(self):
        """
        Rebuild partitions and resource_agenda from current state.
        Also schedule call_me_later at now+timeout if any node state changed.
        """
        self.events = []

        # Reconcile queues (no future phases added here)
        self._rebuild_resource_agenda_global()

        # Rebuild partitions (disjoint)
        self._build_partitions()

        # Optional: call_me_later if state changed
        new_sig = self._snapshot_state_sig()
        state_changed = (
            set(new_sig.keys()) != set(self._old_state_sig.keys())
            or any(new_sig[nid] != self._old_state_sig.get(nid) for nid in new_sig)
        )
        if state_changed and self.timeout:
            self.push_event(self.current_time + self.timeout,
                            {'type': 'call_me_later'})
        self._old_state_sig = new_sig

    # ---------------- Readiness helpers ----------------
    def _node_ready_at(self, node):
        """
        Predict the absolute time when 'node' can start computing if selected now.
        Uses machines_transitions for durations.
        """
        now = self.current_time
        node_id = node['id']
        state = node.get('state')
        started_at = node.get('phase_started_at')

        # needed durations
        t_switching_on_to_active = self._transition_time(
            node_id, 'switching_on',  'active',   0.0)
        t_switching_off_to_sleeping = self._transition_time(
            node_id, 'switching_off', 'sleeping', 0.0)
        t_sleeping_to_active = self._transition_time(
            node_id, 'sleeping',      'active',   0.0)

        ag_entry = self._agenda_by_id().get(node_id)
        head = ag_entry['queue'][0] if ag_entry and ag_entry['queue'] else None

        # Already active & idle: ready now
        if state == 'active' and node.get('job_id') is None:
            return now

        if state == 'switching_on':
            # finish switching_on -> active
            if head and head['phase'] == 'switching_on':
                return float(head['finish_time'])
            if started_at is not None:
                return float(started_at) + t_switching_on_to_active
            return now + t_switching_on_to_active

        if state == 'switching_off':
            # finish switching_off to sleeping, then wake sleeping->active
            if head and head['phase'] == 'switching_off':
                return float(head['finish_time']) + t_sleeping_to_active
            if started_at is not None:
                return float(started_at) + t_switching_off_to_sleeping + t_sleeping_to_active
            return now + t_switching_off_to_sleeping + t_sleeping_to_active

        if state == 'sleeping':
            return now + t_sleeping_to_active

        # conservative default
        return now
