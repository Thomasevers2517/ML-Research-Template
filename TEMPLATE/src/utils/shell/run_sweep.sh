#!/bin/bash

# Check if GPU IDs and project name are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 [project-name] [GPU IDs]"
    exit 1
fi

# Extract project name and shift it from arguments
project=$1
shift 1  # Remove the project name from the argument list

# Generate a random folder name for outputs
output_folder="TEMPLATE/logging/local_output/$(date +%s%N)"
echo "Creating output folder: $output_folder"
mkdir -p "$output_folder"

sweep_config_path="TEMPLATE/configs/training/sweep_config.yaml"
# Run the sweep command and capture both stdout and stderr
echo "Running sweep command: wandb sweep $sweep_config_path"
sweep_output=$(wandb sweep --project $project $sweep_config_path 2>&1)

# Print the sweep output for debugging
echo "Sweep Output:"
echo "$sweep_output"

# Extract the Sweep ID using grep and awk
sweep_id=$(echo "$sweep_output" | grep -Eo "wandb: Creating sweep with ID: [a-z0-9]+" | awk '{print $NF}')

# Check if the sweep ID was extracted successfully
if [ -z "$sweep_id" ]; then
    echo "Error: Failed to extract the sweep ID."
    echo "Full sweep output:"
    echo "$sweep_output"
    exit 1
fi

# Output the Sweep ID and instructions
echo "Extracted Sweep ID: $sweep_id"
echo "View sweep at: https://wandb.ai/thomasevers9/$project/sweeps/$sweep_id"
echo "Run sweep agent with: wandb agent thomasevers9/$project/$sweep_id"

# Start the sweep agent using the provided GPU IDs
echo "Starting sweep agent on GPUs: $@"
for i in $@; do
    log_file="$output_folder/gpu$i.txt"
    echo "Running command: CUDA_VISIBLE_DEVICES=$i nohup wandb agent thomasevers9/$project/$sweep_id > $log_file &"

    CUDA_VISIBLE_DEVICES=$i nohup wandb agent thomasevers9/$project/$sweep_id > "$log_file" &
    sleep 8
done
