
# Train the agent
# the function to run experiments
# data_set_name is one of the following: emotion, snli, sst2
# task_type is one of the following: explain, attack
# features_type is one of the following: const, random, statistical_bin, effective_information, gradient, gradient_input, input_ids, original_embedding, mixture
# token_replacement_strategy is one of the following: mask, delete
# selected_layers is a list of integers, e.g., 0 1 2 3 4 5

run_train() {
    python run_train.py --data_set_name $1 --task_type $2 --use_wandb --features_type $3 --token_replacement_strategy $4 --use_ddqn --max_sampling_steps 2000000
}

# an example of run a single training experiment:
run_train emotion explain const mask
# You can change the parameters in order to run all the experiments, note that the time to run all the experiments can be very long.


# Evaluate the agent
# the function to run experiments
run_eval(){
    python run_eval.py --data_set_name $1 --task_type $2 --features_type $3 --token_replacement_strategy $4 --dqn_weights_path $5
}

# an example of run a single evaluation experiment:
run_eval emotion explain const mask ./saved_weights/emotion_explain_const_mask.pth
