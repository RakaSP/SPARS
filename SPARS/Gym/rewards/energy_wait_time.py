# SPARS/Gym/rewards/energy_wait_time.py
import logging
import torch as T

logger = logging.getLogger("runner")


class Reward:
    """
    reward_per_node = α * (-energy_waste_per_node) + β * (-waiting_time_metric)
    where waiting_time_metric is a scalar (mean wait of finished jobs) broadcast to all nodes.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.9,
                 # mean wait per finished job (True) or sum (False)
                 use_mean_wait: bool = True,
                 device: str = "cuda",
                 require_grad: bool = True):   # keep True to match your current pipeline
        self.alpha = alpha
        self.beta = beta
        self.use_mean_wait = use_mean_wait
        self.device = T.device(device)
        self.require_grad = require_grad

    @staticmethod
    def _get_first_key(d: dict, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _compute_waiting_time_scalar(self, monitor) -> float:
        """
        Compute a scalar waiting-time metric from monitor logs:
        wait_j = max(0, start_time_j - submit_time_j) for each job with both times available.
        Returns 0.0 if no valid pairs found.
        """
        submit_by_id = {}
        for job in monitor.jobs_submission_log:
            jid = self._get_first_key(job, ["job_id"])
            submit = self._get_first_key(
                job, ["subtime"])
            if jid is not None and submit is not None:
                submit_by_id[jid] = float(submit)

        waits = []
        for job in monitor.jobs_execution_log:
            jid = self._get_first_key(job, ["job_id", "id", "jid"])
            start = self._get_first_key(
                job, ["start_time"])
            submit = submit_by_id.get(jid)
            if submit is not None and start is not None:
                wait = float(start) - float(submit)
                if wait > 0:
                    waits.append(wait)

        if not waits:
            return 0.0
        if self.use_mean_wait:
            return float(sum(waits) / len(waits))
        else:
            return float(sum(waits))

    def calculate_reward(self, monitor, waiting_queue, current_time):
        """
        Returns a scalar reward (torch tensor) on self.device.
        Higher energy waste or waiting time => more negative reward.
        """
        total_waste = sum(float(e.get('energy_waste'))
                          for e in monitor.energy)
        total_wait = sum(
            float(current_time) - float(j.get('subtime'))
            for j in waiting_queue
        )

        waste_t = T.tensor(total_waste, dtype=T.float32, device=self.device)
        wait_t = T.tensor(total_wait,  dtype=T.float32, device=self.device)

        penalty = self.alpha * waste_t + self.beta * wait_t
        reward = -penalty  # penalize waste & wait

        logger.info(
            f"total_waste={waste_t.item():.4f}, total_wait={wait_t.item():.4f}, reward={reward.item():.4f}")
        return reward  # 0-D tensor on self.device
