import numpy as np


def feature_extraction(simulator) -> np.ndarray:
    # === GLOBAL FEATURES ===
    num_sim_features = 5
    simulator_features = np.zeros((num_sim_features,), dtype=np.float32)

    job_num = len(simulator.jobs_manager.waiting_queue)
    simulator_features[0] = job_num

    arrival_rate = len(simulator.Monitor.jobs_submission_log) / (
        simulator.current_time - simulator.start_time + 1e-8
    )
    simulator_features[1] = arrival_rate

    mean_runtime_jobs_in_queue = sum(
        job["runtime"] for job in simulator.jobs_manager.waiting_queue
    ) / (len(simulator.jobs_manager.waiting_queue) + 1e-8)
    simulator_features[2] = mean_runtime_jobs_in_queue

    total_energy_waste = sum(e["energy_waste"]
                             for e in simulator.Monitor.energy)
    simulator_features[3] = total_energy_waste

    mean_requested_walltime_jobs_in_queue = mean_runtime_jobs_in_queue
    simulator_features[4] = mean_requested_walltime_jobs_in_queue

    # === NODE FEATURES ===
    num_node_features = 6
    hosts = list(simulator.PlatformControl.get_state())
    num_nodes = len(hosts)

    node_rows = []
    for i in range(num_nodes):
        node_features = np.zeros((num_node_features,), dtype=np.float32)
        node_state = hosts[i].get("state")
        node_job_id = hosts[i].get("job_id")

        if node_state == 'sleeping':
            node_features[0] = 1
        elif node_state == 'switching_off':
            node_features[0] = 2
        elif node_state == 'switching_on':
            node_features[0] = 3
        elif node_state == 'active':
            node_features[0] = 4

        if node_state == 'active' and node_job_id is not None:
            node_features[1] = 1
        else:
            node_features[1] = 0

        for _node_state in simulator.Monitor.nodes_state:
            if _node_state['id'] == i:
                if _node_state['state'] == 'active' and _node_state['job_id'] is None:
                    node_features[2] = _node_state['duration']
                else:
                    node_features[2] = 0
                current_state_duration = _node_state['duration']
                break

        release_by_id = simulator.scheduler.algorithm._releases_by_id()
        node_features[3] = release_by_id[i].get('release_time')

        for energy in simulator.Monitor.energy:
            if energy['id'] == i:
                node_features[4] = energy['energy_waste']
                break

        state_hist = simulator.Monitor.states_hist_by_id[i]
        switch_on_duration = 0
        switch_off_duration = 0
        for state_entry in state_hist['state_history']:
            if state_entry['state'] == 'switching_on':
                switch_on_duration += state_entry['finish_time'] - \
                    state_entry['start_time']
            elif state_entry['state'] == 'switching_off':
                switch_off_duration += state_entry['finish_time'] - \
                    state_entry['start_time']

        node_features[5] = switch_on_duration + switch_off_duration
        if node_features[0] in (2, 3):
            node_features[5] += current_state_duration

        node_rows.append(node_features)

    # stack per-node rows -> (num_nodes, 6)
    node_features_mat = np.vstack(node_rows).astype(np.float32)

    # broadcast global -> (num_nodes, 5)
    sim_broadcast = np.broadcast_to(
        simulator_features, (num_nodes, simulator_features.shape[0]))

    # concat -> (num_nodes, 11)
    features = np.concatenate(
        (sim_broadcast, node_features_mat), axis=1).astype(np.float32)
    return features
