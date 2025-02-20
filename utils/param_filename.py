def get_param_filename(reward_params, penalty_params, seed, train_max_steps):
        reward_str = f"newobj_{reward_params['new_object']}_explr_{reward_params['tot_ep_exploration']}"
        penalty_str = f"samepos_{penalty_params['same_position']}_obstacle_{penalty_params['obstacle']}"
        max_steps_str = f"steps_{train_max_steps}"
        seed_str = f"seed_{seed}"
        return f"{reward_str}_{penalty_str}_{max_steps_str}_{seed_str}"