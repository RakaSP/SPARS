from operator import itemgetter


class BaseAlgorithm:
    """
    Per-node Next Releases: Store queue of event to get the earliest node's next idle state
      next_releases = [
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
        self.machines_transitions = machines.machines_transition
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
        self.next_releases = [
            {'node_id': n['id'], 'queue': [],
                'release_time': self.current_time}
            for n in self.state
        ]

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
    def _releases_by_id(self):
        return {e['node_id']: e for e in self.next_releases}

    @staticmethod
    def _sum_queue_abs(q):
        """Return the absolute finish time of last phase or 0 if empty (caller sets to now)."""
        return float(q[-1]['finish_time'])

    def _remaining_time(self, total, started_at, now):
        """Remaining time in current phase; conservative if timestamps unknown."""

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
        """Where the next phase would start: end of last phase, else entry['release_time'] (can be 0.0)."""
        q = entry['queue']
        if q:
            return float(q[-1]['finish_time'])
        # queue empty -> use recorded release_time (0.0 is allowed by your policy)
        return float(entry['release_time'])

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
        source = getattr(self, "machines_transitions")
        for row in source:
            nid = row.get("node_id")
            tlist = row.get("transitions") or []
            by_pair = {}
            for t in tlist:
                frm = str(t.get("from"))
                to = str(t.get("to"))
                tt = float(t.get("transition_time"))
                by_pair[(frm, to)] = tt
            if nid is not None:
                self._trans_index[int(nid)] = by_pair

        self._trans_index_built = True

    def _transition_time(self, node_id: int, from_state: str, to_state: str) -> float:
        """Return transition_time for (from_state -> to_state) for node_id; default if not found."""
        self._ensure_transition_index()
        by_pair = self._trans_index.get(int(node_id))
        return float(by_pair.get((from_state, to_state)))

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

        if q and q[0]['phase'] == phase_name:
            if start_at is not None:
                q[0]['start_time'] = float(start_at)
                q[0]['finish_time'] = float(start_at) + float(duration)
        else:
            st = float(start_at)
            ft = st + float(duration)
            q.insert(0, {'phase': phase_name,
                     'start_time': st, 'finish_time': ft})

    # Rebuild to “earliest idle”
    def _rebuild_next_releases_global(self):
        by_id = self._releases_by_id()
        now = self.current_time

        for node in self.state:
            nid = node['id']
            entry = by_id.get(nid)
            if entry is None:
                entry = {'node_id': nid, 'queue': [], 'release_time': now}
                self.next_releases.append(entry)

            # Drop phases already finished
            if entry['queue']:
                entry['queue'] = [seg for seg in entry['queue']
                                  if float(seg['finish_time']) > now]

            state = node['state']
            job_id = node.get('job_id')

            # Transition durations
            t_off_sleep = self._transition_time(
                nid, 'switching_off',  'sleeping')
            t_sleep_on = self._transition_time(
                nid, 'sleeping',       'switching_on')
            t_on_active = self._transition_time(
                nid, 'switching_on',   'active')

            q = entry['queue']
            head = q[0] if q else None
            head_phase = head['phase'] if head else None

            if state == 'switching_off':
                # Ensure head switching_off; reuse its start_time if already present
                if head_phase != 'switching_off':
                    q.insert(0, {'phase': 'switching_off',
                                 'start_time': now, 'finish_time': now + t_off_sleep})
                    head = q[0]
                cursor = float(head['finish_time'])
                # Immediately proceed: sleeping -> switching_on (instant/short) -> active
                start_on = cursor + float(t_sleep_on)
                self._append_phase_abs(entry, 'switching_on',
                                       start_on, t_on_active)

            elif state == 'sleeping':
                # sleeping -> switching_on (instant/short) -> active
                start_on = now + float(t_sleep_on)
                self._append_phase_abs(entry, 'switching_on',
                                       start_on, t_on_active)

            elif state == 'switching_on':
                # Ensure head switching_on; reuse its start_time if already present
                if head_phase != 'switching_on':
                    q.insert(0, {'phase': 'switching_on',
                                 'start_time': now, 'finish_time': now + t_on_active})
                else:
                    # normalize finish in case duration changed
                    head['finish_time'] = float(
                        head['start_time']) + float(t_on_active)

            elif state == 'active':
                if job_id is None:
                    # active & idle: strip stray switching heads (we're already idle)
                    while q and q[0]['phase'] in ('switching_off', 'switching_on'):
                        q.pop(0)
                # if computing, keep existing compute phases (allocator added them)

            # release_time = end of last phase, or now if none
            self._recompute_release_at(entry)

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
                if (node.get('state') != 'active') or (node.get('job_id') is not None):
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
                             if node_by_id.get(nid).get('state') == 'sleeping']
        if sleeping_reserved:
            self.push_event(self.current_time, {
                'type': 'switch_on',
                'nodes': sleeping_reserved
            })

    # ---------------- Allocation ----------------
    def allocate(self, job, allocated_nodes):
        """
        Reserve nodes and append ONLY the compute phase into next_releases.
        Wake/transition phases are already captured by next_releases' release_time.
        """
        if not allocated_nodes:
            return

        # 1) Update partitions
        node_ids = [n['id'] for n in allocated_nodes]
        ids = set(node_ids)
        def _filter_out(lst): return [n for n in lst if n['id'] not in ids]
        self.idle = _filter_out(self.idle)
        self.sleeping = _filter_out(self.sleeping)
        self.switching_on = _filter_out(self.switching_on)
        self.switching_off = _filter_out(self.switching_off)
        self.reserved.extend(allocated_nodes)

        # 2) Register with jobs_manager
        self.jobs_manager.add_job_to_scheduled_queue(job['job_id'], node_ids)

        # 3) Compute walltime via slowest node
        compute_speed = min(float(n['compute_speed']) for n in allocated_nodes)
        assert compute_speed > 0.0
        walltime = float(job['runtime']) / compute_speed

        # 4) Append ONLY compute at each node's earliest-ready time (release_time)
        by_id = self._releases_by_id()
        for n in allocated_nodes:
            entry = by_id.get(n['id'])
            # next_releases should already exist from prep_schedule()
            assert entry is not None, f"next_releases entry missing for node {n['id']}"
            # earliest time node is ACTIVE & IDLE
            cursor = float(entry['release_time'])
            self._append_phase_abs(
                entry, f'compute(job={job["job_id"]})', cursor, walltime)

    # ---------------- Timeout handling ----------------

    def remove_from_timeout_list(self, node_ids):
        ids = set(node_ids)
        self.timeout_list[:] = [
            ti for ti in self.timeout_list if ti.get('node_id') not in ids]

    def _rebuild_timeout_list(self):
        """
        Recompute timeout_list from CURRENT state/partitions.
        Policy:
        - If timeout is None: no timeouts -> clear list & marker.
        - Else: every ACTIVE & IDLE, NON-RESERVED node must have a deadline.
                Non-idle or reserved nodes must not have a deadline.
        """
        if self.timeout is None:
            self.timeout_list = []
            self.next_timeout_at = None
            return

        now = self.current_time
        expire_at = now + self.timeout

        # Build fast lookups
        reserved_ids = {n['id'] for n in self.reserved}
        idle_ids = {
            n['id'] for n in self.state
            if (n.get('state') == 'active') and (n.get('job_id') is None)
        }

        # Keep only entries for currently idle & not-reserved nodes
        keep_map = {}
        for t in self.timeout_list:
            nid = t['node_id']
            # keep only valid entries; strict access to 'time'
            if (nid in idle_ids) and (nid not in reserved_ids):
                keep_map[nid] = float(t['time'])

        # Ensure every eligible node has a deadline; assign new ones to now+timeout
        for nid in (idle_ids - reserved_ids):
            if nid not in keep_map:
                keep_map[nid] = expire_at

        # Write back as a list (unsorted is fine; timeout_policy will derive next_earliest)
        self.timeout_list = [{'node_id': nid, 'time': t}
                             for nid, t in keep_map.items()]

    def timeout_policy(self):
        if self.timeout is None:
            return

        now = self.current_time

        # NEW: refresh (adds new idle nodes, removes reserved/non-idle)
        self._rebuild_timeout_list()

        state_by_id = {n['id']: n for n in self.state}
        reserved_ids = {n['id'] for n in self.reserved}

        keep, switch_off, next_earliest = [], [], None
        for t in self.timeout_list:
            nid = t['node_id']
            node = state_by_id.get(nid)
            if node is None:
                continue
            if nid in reserved_ids:
                continue
            idle = (node.get('state') == 'active') and (
                node.get('job_id') is None)
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

        if next_earliest is not None and self.next_timeout_at != next_earliest:

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
        """
        self.events = []

        # Reconcile queues (no future phases added here)
        self._rebuild_next_releases_global()

        # Rebuild partitions (disjoint)
        self._build_partitions()

        # Rebuild timeout_list
        self._rebuild_timeout_list()

    # ---------------- Readiness helpers ----------------
    def _node_ready_at(self, node):
        """
        Predict the absolute time when 'node' can start computing if selected now.
        This simply returns the 'release_time' from next_releases.
        """
        # Fetch the release time directly from next_releases (which holds the calculated next event time)
        node_id = node['id']
        entry = self._releases_by_id().get(node_id)

        if entry:
            return entry['release_time']

        # If no entry found, return current time as fallback (or handle as error)
        return self.current_time
