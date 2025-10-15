import logging
from typing import Dict, Any
import torch as T

logger = logging.getLogger("runner")


class Reward:
    """
    reward_per_node = α * (-energy_waste_term) + β * (-waiting_time_term)

    If use_mean_wait=True, the waiting-time term uses the MEAN wait of the jobs
    that started in THIS step (next_monitor vs monitor). Otherwise it uses SUM.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.9,
        use_mean_wait: bool = True,
        device: str = "cuda",
        require_grad: bool = True,
        # Δt (used in normalization), was 1800 literal
        tick_seconds: float = 1800.0,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_mean_wait = bool(use_mean_wait)
        self.device = T.device(device)
        self.require_grad = bool(require_grad)
        self.tick_seconds = float(tick_seconds)

    # --------------------------
    # Helpers
    # --------------------------
    def _to_tensor(self, value: float) -> T.Tensor:
        return T.tensor(value, dtype=T.float32, device=self.device, requires_grad=self.require_grad)

    @staticmethod
    def _sum_wait(logs: list[Dict[str, Any]]) -> float:
        # Robust to missing keys/None
        total = 0.0
        for log in logs:
            try:
                st = log["start_time"]
                sub = log["subtime"]
                if st is not None and sub is not None:
                    total += float(st - sub)
            except Exception:
                # Ignore malformed entries but keep training running
                continue
        return total

    # --------------------------
    # Terms
    # --------------------------
    def wasted_energy_reward(self, monitor, next_monitor) -> T.Tensor:
        """
        R1 = (next_total_waste - current_total_waste) normalized by total ECR * Δt
        Assumes each node is ACTIVE: uses its dvfs_mode to fetch ECR.
        """
        current_total_waste = self._sum_wait([{"start_time": e.get(
            "energy_waste"), "subtime": 0.0} for e in monitor.energy])
        next_total_waste = self._sum_wait([{"start_time": e.get(
            "energy_waste"), "subtime": 0.0} for e in next_monitor.energy])
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

    def waiting_time_reward(self, monitor, next_monitor, waiting_queue) -> T.Tensor:
        """
        Step waiting-time change:
          step_total_wt = (Σ wait in next) - (Σ wait in current)
        If use_mean_wait=True, divide by #newly-started jobs in this step.

        Normalization:
          - mean mode  : divide by Δt
          - sum mode   : divide by (#waiting) * Δt
        """
        current_total_wt = self._sum_wait(monitor.jobs_submission_log)
        next_total_wt = self._sum_wait(next_monitor.jobs_submission_log)
        step_total_wt = next_total_wt - current_total_wt

        # how many NEW jobs started this step (log length increase)
        new_started = len(getattr(next_monitor, "jobs_submission_log")
                          ) - len(getattr(monitor, "jobs_submission_log"))

        if self.use_mean_wait and new_started > 0:
            metric = step_total_wt / new_started
            denom = max(self.tick_seconds, 1e-9)
        else:
            metric = step_total_wt

            total_not_executed = len(waiting_queue)
            denom = max(total_not_executed * self.tick_seconds, 1e-9)

        normalized_R2 = -self.beta * (metric / denom)
        return self._to_tensor(normalized_R2)

    def calculate_reward(self, monitor, next_monitor, waiting_queue) -> T.Tensor:
        return self.wasted_energy_reward(monitor, next_monitor) + \
            self.waiting_time_reward(
                monitor, next_monitor, waiting_queue)
