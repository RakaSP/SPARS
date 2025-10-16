# gym_env.py
import copy
import numpy as np
import gymnasium as gym
import logging
import torch as T

# import your real Simulator and RJMS
from SPARS.Gym.utils import Reward, action_translator, get_feasible_mask
from SPARS.Simulator.Simulator import Simulator

from SPARS.Gym.utils import feature_extraction

CPU_DEVICE = T.device("cpu")

logger = logging.getLogger("runner")


class HPCGymEnv(gym.Env):
    """
    Gymnasium environment that ONLY wraps Simulator + RJMS.
    Responsibilities:
      - advance_system(): run sim -> rjms -> apply rjms events -> return features (pre-action)
      - apply_action(action): translate agent action -> apply to sim -> compute reward (prev vs next)
      - step(action): helper for Gym compatibility -> advance_system + apply_action
    No agent/critic/memory inside.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, simulator, device=CPU_DEVICE):
        super().__init__()

        self.simulator = simulator
        self.device = device

    def advance(self):
        need_rl = False
        while not need_rl and self.simulator.is_running:
            events = self.simulator.proceed()
            scheduler_message = self.simulator.scheduler.schedule(self.simulator.current_time, self.simulator.PlatformControl.get_state(
            ), self.simulator.jobs_manager.waiting_queue, self.simulator.jobs_manager.scheduled_queue)

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                _events = _data['events']
                for event in _events:
                    self.simulator.push_event(timestamp, event)

            for event_list in events['event_list']:
                for event in event_list['events']:
                    if event['type'] == 'CALL_RL':
                        need_rl = True
                        break
                if need_rl:
                    break

        observation = self.get_observation()

        return observation

    def step(self, actions):
        state = self.simulator.PlatformControl.get_state()
        logger.info(f"Action taken: {actions}")

        """"Action translator for Scalar Active Target"""
        # rl_events = action_translator(
        #     self.simulator.Monitor.num_nodes, state, actions, self.simulator.current_time)

        """Thomas Action Translator"""
        monitor = copy.deepcopy(self.simulator.Monitor)
        rl_events = action_translator(
            actions, state, self.simulator.current_time)

        for _rl_event in rl_events:
            self.simulator.push_event(
                timestamp=_rl_event['time'], event=_rl_event['event'])

        need_rl = False

        prev_current_time = self.simulator.current_time

        while not need_rl and self.simulator.is_running:
            events = self.simulator.proceed()

            for event_list in events['event_list']:
                for event in event_list['events']:
                    if event['type'] == 'CALL_RL':
                        need_rl = True
                        break
                if need_rl:
                    break

            scheduler_message = self.simulator.scheduler.schedule(
                self.simulator.current_time)

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                _events = _data['events']
                for event in _events:
                    self.simulator.push_event(timestamp, event)

            for event_list in events['event_list']:
                for event in event_list['events']:
                    if event['type'] == 'CALL_RL':
                        need_rl = True
                        break
                if need_rl:
                    break

        next_current_time = self.simulator.current_time
        reward_function = Reward()

        """SPARS Calculate Reward"""
        # reward = reward_function.calculate_reward(
        #     self.simulator.Monitor, self.simulator.jobs_manager.waiting_queue, self.simulator.current_time)

        """Thomas Calculate Reward"""
        future_monitor = self.simulator.Monitor
        reward = reward_function.calculate_reward(
            monitor, future_monitor, self.simulator.jobs_manager.waiting_queue, prev_current_time, next_current_time)
        done = not self.simulator.is_running
        observation = self.get_observation()

        return observation, reward, done

    def reset(self, simulator):
        self.simulator = simulator

    def get_observation(self):
        num_nodes = self.simulator.Monitor.num_nodes

        states = self.simulator.PlatformControl.get_state()
        mask = get_feasible_mask(states)
        features = feature_extraction(self.simulator)

        features_ = T.from_numpy(features).to(self.device).float()

        mask = np.asanyarray(mask)
        mask = mask.reshape(1, num_nodes, 2)
        mask_ = T.from_numpy(mask).to(self.device).float()

        observation = (features_, mask_)

        return observation
