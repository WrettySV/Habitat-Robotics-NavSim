def get_param_filename(env, train_max_steps):
        reward_str = f"newobj_{env.reward_params['new_object']}_explr_{env.reward_params['tot_ep_exploration']}"
        penalty_str = f"samepos_{env.penalty_params['same_position']}_obstacle_{env.penalty_params['obstacle']}"
        max_steps_str = f"steps_{train_max_steps}"
        seed_str = f"seed_{env.seed}"
        return f"{reward_str}_{penalty_str}_{max_steps_str}_{seed_str}"