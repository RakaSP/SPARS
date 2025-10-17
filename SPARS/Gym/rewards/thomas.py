from SPARS.Utils import get_global_logger
from typing import Dict, Any
import torch as T

logger = get_global_logger()


class Reward:
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.9,
        device: str = "cuda",
        require_grad: bool = True,
        # Δt (used in normalization), was 1800 literal
        tick_seconds: float = 1800.0,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.device = T.device(device)
        self.require_grad = bool(require_grad)
        self.tick_seconds = float(tick_seconds)

    # --------------------------
    # Helpers
    # --------------------------
    def _to_tensor(self, value: float) -> T.Tensor:
        return T.tensor(value, dtype=T.float32, device=self.device, requires_grad=self.require_grad)

    @staticmethod
    def _sum_wait(logs: list[Dict[str, Any]], time) -> float:
        # Robust to missing keys/None
        total = 0.0
        for log in logs:
            sub = log["subtime"]
            total += (time - sub)

        return total

    # --------------------------
    # Terms
    # --------------------------
    def wasted_energy_reward(self, monitor, next_monitor) -> T.Tensor:
        """
        R1 = (next_total_waste - current_total_waste) normalized by total ECR * Δt
        Assumes each node is ACTIVE: uses its dvfs_mode to fetch ECR.
        """
        current_total_waste = sum(e.get('energy_waste')
                                  for e in monitor.energy)
        next_total_waste = sum(e.get('energy_waste')
                               for e in next_monitor.energy)
        R1 = next_total_waste - current_total_waste

        # Build index: node_id -> dvfs_profiles
        ecr_by_id: Dict[int, Dict[str, float]] = {
            e["id"]: e["dvfs_profiles"] for e in monitor.ecr}

        # Total ECR assuming nodes are active ⇒ use dvfs profile for each node's dvfs_mode
        # This will raise KeyError on unknown id/mode (prefer loud fail over silent 0).
        total_ecr = 0.0
        for n in monitor.nodes_state:
            total_ecr += float(ecr_by_id[n["id"]][n["dvfs_mode"]])

        denom = max(total_ecr * self.tick_seconds, 1e-9)  # avoid div/0
        normalized_R1 = -self.alpha * (R1 / denom)
        return self._to_tensor(normalized_R1)

    def waiting_time_reward(self, next_monitor, waiting_queue, current_time, next_time) -> T.Tensor:

        total_waiting_time = 0
        max_total_waiting_time = 0

        jobs_submission_log = next_monitor.jobs_submission_log
        jobs_submitted_ids = {job["job_id"] for job in jobs_submission_log}
        for job in jobs_submission_log:
            if current_time < job["start_time"] <= next_time:
                total_waiting_time += (job["start_time"] -
                                       max(job['subtime'], current_time))
                max_total_waiting_time += (next_time -
                                           max(job['subtime'], current_time))

        jobs_arrival_log = next_monitor.jobs_arrival_log

        for job in jobs_arrival_log:
            if job['job_id'] not in jobs_submitted_ids:
                total_waiting_time += (next_time -
                                       max(job['subtime'], current_time))
                max_total_waiting_time += (next_time -
                                           max(job['subtime'], current_time))
        if max_total_waiting_time > 0:
            R2 = total_waiting_time / max_total_waiting_time
        else:
            R2 = 0.0

        return self._to_tensor(-self.beta * R2)

    def calculate_reward(self, monitor, next_monitor, waiting_queue, current_time, next_time) -> T.Tensor:
        return self.wasted_energy_reward(monitor, next_monitor) + \
            self.waiting_time_reward(
                next_monitor, waiting_queue, current_time, next_time)
