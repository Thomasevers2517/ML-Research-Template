#!/bin/bash

# Description: This script finds and kills all processes with 'agent' in the name and GPU processes owned by 'thomasevers'.

# Get the username
USER_TO_KILL="thomasevers"

echo "The following processes with 'agent' in the name owned by '$USER_TO_KILL' were found:"
echo "-----------------------------------------------------------"

# Find all PIDs of processes with 'agent' in their command line owned by 'thomasevers'
agent_pids=$(pgrep -u "$USER_TO_KILL" -f 'agent')

if [ -z "$agent_pids" ]; then
    echo "No processes with 'agent' in the name owned by '$USER_TO_KILL' were found."
else
    # Display detailed information about each process
    for pid in $agent_pids; do
        ps -fp "$pid"
    done
fi

echo "-----------------------------------------------------------"

echo "The following GPU processes owned by '$USER_TO_KILL' are running:"

# Check for nvidia-smi command
if ! command -v nvidia-smi &> /dev/null
then
    echo "Error: 'nvidia-smi' command not found. Please ensure NVIDIA drivers are installed and 'nvidia-smi' is in your PATH."
    exit 1
fi

echo "-----------------------------------------------------------"

# Get the list of PIDs of GPU processes owned by 'thomasevers'
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

# Remove any empty lines
gpu_pids=$(echo "$gpu_pids" | sed '/^\s*$/d')

# Initialize an array to hold PIDs of GPU processes owned by the user
declare -a user_gpu_pids=()

# Loop through GPU PIDs and filter by user
for pid in $gpu_pids; do
    pid_user=$(ps -o user= -p "$pid" 2>/dev/null)
    if [ "$pid_user" = "$USER_TO_KILL" ]; then
        user_gpu_pids+=("$pid")
    fi
done

if [ ${#user_gpu_pids[@]} -eq 0 ]; then
    echo "No GPU processes owned by '$USER_TO_KILL' were found."
else
    # Display detailed information about each process
    for pid in "${user_gpu_pids[@]}"; do
        ps -fp "$pid"
    done
fi

echo "-----------------------------------------------------------"

# Check if there are any processes to kill
if [ -z "$agent_pids" ] && [ ${#user_gpu_pids[@]} -eq 0 ]; then
    echo "No processes to kill."
    exit 0
fi

# Prompt the user for confirmation
read -p "Do you want to kill these processes? [y/N]: " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    # Kill 'agent' processes
    if [ -n "$agent_pids" ]; then
        echo "Killing 'agent' processes..."
        kill -15 $agent_pids
        kill -9 $agent_pids
    fi

    # Kill GPU processes
    if [ ${#user_gpu_pids[@]} -gt 0 ]; then
        echo "Killing GPU processes..."
        kill -15 "${user_gpu_pids[@]}"
    fi

    echo "Processes have been signaled to terminate."

    # Optionally, wait and check if processes have terminated
    sleep 2

    # Check if any 'agent' processes are still running
    remaining_agent_pids=()
    for pid in $agent_pids; do
        if ps -p "$pid" > /dev/null 2>&1; then
            remaining_agent_pids+=("$pid")
        fi
    done

    # Check if any GPU processes are still running
    remaining_gpu_pids=()
    for pid in "${user_gpu_pids[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            remaining_gpu_pids+=("$pid")
        fi
    done

    # Force kill any remaining processes
    if [ ${#remaining_agent_pids[@]} -gt 0 ] || [ ${#remaining_gpu_pids[@]} -gt 0 ]; then
        echo "Some processes did not terminate. Attempting to force kill..."
        kill -9 "${remaining_agent_pids[@]}" "${remaining_gpu_pids[@]}"
        echo "Forcefully killed remaining processes."
    else
        echo "All processes terminated successfully."
    fi
else
    echo "No processes were killed."
fi
