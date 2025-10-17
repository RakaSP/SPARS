# SPARS/Gym/translators/scalar_active_target.py
# NOTE: Logic unchanged; comments added only.
from SPARS.Utils import get_global_logger
import torch as T

logger = get_global_logger()


def action_translator(logits, state, current_time):
    """
    Args:
      logits: [N,2] or [1,N,2] tensor/array/list
              left  -> switch_off score
              right -> switch_on  score
      state : list of per-node dicts with keys: 'state', 'job_id'
    Returns:
      events: [{'time':..., 'event': {'type': 'switch_off', 'nodes': [...]}},
               {'time':..., 'event': {'type': 'switch_on',  'nodes': [...]}}]
               (omitted if empty)
    """

    # Align lengths if state/logits mismatch
    switch_off = []
    switch_on = []

    for idx, value in enumerate(logits):
        if value <= 0.5:
            switch_off.append(idx)
        else:
            switch_on.append(idx)

    # Build current sets
    current_idle = []                # active & job_id is None
    current_sleeping = []            # state == 'sleeping'
    # (active & job_id != None) OR switching_{on,off}
    unable_to_make_action = []

    for i, n in enumerate(state):
        st = str(n.get('state', '')).lower()
        jid = n.get('job_id', None)

        if st == 'active' and jid is None:
            current_idle.append(i)
        if st == 'sleeping':
            current_sleeping.append(i)
        if (st == 'active' and jid is not None) or (st in ('switching_on', 'switching_off')):
            unable_to_make_action.append(i)

    unable = set(unable_to_make_action)
    idle_set = set(current_idle)
    sleeping_set = set(current_sleeping)

    # Apply filters:
    # - ignore any node in 'unable'
    # - switch_on only if currently sleeping
    # - switch_off only if currently idle
    switch_on = [i for i in switch_on if i not in unable and i in sleeping_set]
    switch_off = [i for i in switch_off if i not in unable and i in idle_set]

    events = []
    if switch_off:
        events.append({'time': current_time, 'event': {
                      'type': 'switch_off', 'nodes': switch_off}})
    if switch_on:
        events.append({'time': current_time, 'event': {
                      'type': 'switch_on',  'nodes': switch_on}})

    return events
