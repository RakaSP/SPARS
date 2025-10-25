# SPARS/Gym/translators/scalar_active_target.py
from SPARS.Utils import get_global_logger
import torch as T

logger = get_global_logger()

def action_translator(actions, state, current_time):
    """
    Translates discrete actions into switch_on/off events.

    Args:
      actions: tensor/list of shape [N] or [1, N] or [N, 1]
               Each entry is 0 (request switch_off) or 1 (request switch_on)
      state: list of per-node dicts with keys: 'state', 'job_id'
      current_time: simulation time

    Returns:
      events: list of dicts like:
          [{'time': t, 'event': {'type': 'switch_on',  'nodes': [...] }},
           {'time': t, 'event': {'type': 'switch_off', 'nodes': [...] }}]
    """
    x = T.as_tensor(actions, dtype=T.int64)
    if x.dim() > 1:
        x = x.squeeze()
    if x.dim() != 1:
        raise ValueError(f"Expected action vector [N] or [1,N], got {tuple(x.shape)}")

    N = x.size(0)
    if len(state) != N:
        M = min(N, len(state))
        x = x[:M]
        state = state[:M]
        N = M

    # Classify which nodes *could* take an action
    current_idle = []       # active & job_id is None
    current_sleeping = []   # state == 'sleeping'
    unable = set()          # busy or switching

    for i, n in enumerate(state):
        st = str(n.get('state')).lower()
        jid = n.get('job_id')

        if st == 'active' and jid is None:
            current_idle.append(i)
        elif st == 'sleeping':
            current_sleeping.append(i)
        elif st in ('switching_on', 'switching_off') or (st == 'active' and jid is not None):
            unable.add(i)

    idle_set = set(current_idle)
    sleeping_set = set(current_sleeping)

    # Actions are direct now:
    # 1 → request switch_on (if currently sleeping)
    # 0 → request switch_off (if currently idle)
    want_on_idx  = (x == 1).nonzero(as_tuple=False).squeeze(1).cpu().tolist()
    want_off_idx = (x == 0).nonzero(as_tuple=False).squeeze(1).cpu().tolist()

    switch_on  = [i for i in want_on_idx  if i not in unable and i in sleeping_set]
    switch_off = [i for i in want_off_idx if i not in unable and i in idle_set]

    events = []
    if switch_off:
        events.append({'time': current_time, 'event': {'type': 'switch_off', 'nodes': switch_off}})
    if switch_on:
        events.append({'time': current_time, 'event': {'type': 'switch_on',  'nodes': switch_on}})

    return events
