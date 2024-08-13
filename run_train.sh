#!/bin/bash

# Pre-training phase
python train.py --training_name train_results --training_parameters training_parameters --start_index 0 --end_index 45

# Run all experiments 
start_index=45
increment=45
final_index=405  # Corrected the variable name from end_index to final_index
nb_jobs=3

# Loop through the ranges
while [ $start_index -lt $final_index ]; do
    # Run nb_jobs commands in parallel
    for i in $(seq 1 $nb_jobs); do
        if [ $start_index -ge $final_index ]; then
            break
        fi
        
        end_index=$((start_index + increment))
        python train.py --training_name train_results --training_parameters training_parameters --start_index $start_index --end_index $end_index &

        # Update start_index for the next iteration
        start_index=$end_index
    done
    
    # Wait for the current batch of jobs to complete
    wait
done