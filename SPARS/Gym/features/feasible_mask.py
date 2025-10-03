# SPARS/Gym/features/feasible_mask.py
# NOTE: Logic unchanged; added comments only.
import numpy as np


def get_feasible_mask(states):
    # fm[:, 0] = dibiarkan, dummy
    # fm[:, 1] = boleh matikan/tidak
    # fm[:, 2] = boleh hidupkan/tidak
    feasible_mask = np.ones((len(states), 3), dtype=np.float32)
    is_switching_off = np.asarray(
        [host['state'] == 'switching_off' for host in states])
    is_switching_on = np.asarray(
        [host['state'] == 'switching_on' for host in states])
    is_switching = np.logical_or(is_switching_off, is_switching_on)
    is_idle = np.asarray(
        [host['state'] == 'active' and host['job_id'] is None for host in states])
    is_sleeping = np.asarray([host['state'] == 'sleeping' for host in states])
    is_allocated = np.asarray(
        [host['state'] == 'active' and host['job_id'] is None for host in states])

    # can it be switched off
    is_really_idle = np.logical_and(is_idle, np.logical_not(is_allocated))
    feasible_mask[:, 1] = np.logical_and(
        np.logical_not(is_switching), is_really_idle)

    # can it be switched on
    feasible_mask[:, 2] = np.logical_and(
        np.logical_not(is_switching), is_sleeping)

    # return cuma 2 action, update 15-09-2022
    return feasible_mask[:, 1:]
