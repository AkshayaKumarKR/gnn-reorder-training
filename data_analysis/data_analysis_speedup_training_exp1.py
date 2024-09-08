import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Base directory and settings for the experiments
base_dir = '../experiments/Experiment 1 - Hidden Dimensions'

sets = {
    'Set 1 - Hidden Dimensions = 16': 'dgl',
    'Set 2 - Hidden Dimensions = 64': 'dgl',
    'Set 3 - Hidden Dimensions = 512': 'dgl'
}

# Function to extract training time, reordering strategy, and model type from a file
def extract_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        training_time = data['Training'][0]  # Assuming training time is the first item in the list
        reordering_strategy = data['reordering_strategy']  # Extract the reordering strategy from the file
        model_type = data['model']  # Extract the model type (e.g., 'GCN' or 'GRAPHSAGE')
    return model_type, reordering_strategy, training_time

# Initialize results dictionary
results = {dim: {'GCN': {}, 'GRAPHSAGE': {}} for dim in sets.keys()}

# Loop through each set and extract training times and strategies
for set_name, folder in sets.items():
    set_dir = os.path.join(base_dir, set_name, folder)
    for file_name in os.listdir(set_dir):
        file_path = os.path.join(set_dir, file_name)
        if file_name.endswith('.metrics'):  # Ensure it's a metrics file
            model_type, reordering_strategy, training_time = extract_data(file_path)

            # Assign training times to the corresponding dictionary
            if reordering_strategy not in results[set_name][model_type]:
                results[set_name][model_type][reordering_strategy] = []
            results[set_name][model_type][reordering_strategy].append(training_time)

# Compute the average training time for each strategy per model type
averaged_results = {dim: {'GCN': {}, 'GRAPHSAGE': {}} for dim in sets.keys()}

# Calculate the averages for GCN and GRAPHSAGE separately
for dim, models in results.items():
    for model_type, strategies in models.items():
        for strategy, times in strategies.items():
            if times:  # Ensure there's at least one training time recorded
                averaged_results[dim][model_type][strategy] = np.mean(times)

# Initialize the combined results dictionary
combined_averaged_results = {dim: {} for dim in sets.keys()}

# Calculate the combined averages for each strategy across both models
for dim in sets.keys():
    for strategy in averaged_results[dim]['GCN']:
        # Calculate the average of GCN and GRAPHSAGE for each strategy
        gcn_time = averaged_results[dim]['GCN'].get(strategy, 0)
        graphsage_time = averaged_results[dim]['GRAPHSAGE'].get(strategy, 0)
        combined_averaged_results[dim][strategy] = np.mean([gcn_time, graphsage_time])

# Calculate the average time for random strategies (rand-0, rand-1, rand-2)
average_random_time = {}
for dim in sets.keys():
    random_times = [combined_averaged_results[dim].get(strategy, 0) for strategy in ['rand-0', 'rand-1', 'rand-2']]
    average_random_time[dim] = np.mean(random_times)

# Calculate speedup for each strategy with respect to the average random time
speedup_results = {dim: {} for dim in sets.keys()}
for dim in sets.keys():
    for strategy, time in combined_averaged_results[dim].items():
        if strategy not in ['rand-0', 'rand-1', 'rand-2']:
            speedup_results[dim][strategy] = average_random_time[dim] / time

print("Average Random Times:", average_random_time)
print("Speedup Results:", speedup_results)

# Define the dimensions for plotting
dimensions = {'16': 'Set 1 - Hidden Dimensions = 16', '64': 'Set 2 - Hidden Dimensions = 64', '512': 'Set 3 - Hidden Dimensions = 512'}
strategies = sorted(speedup_results[dimensions['16']].keys())

x = np.arange(len(strategies))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Plotting the bars for speedup for each hidden dimension
rects1 = ax.bar(x - width, [speedup_results[dimensions['16']][key] for key in strategies], width, label='Hidden Dim = 16')
rects2 = ax.bar(x, [speedup_results[dimensions['64']][key] for key in strategies], width, label='Hidden Dim = 64')
rects3 = ax.bar(x + width, [speedup_results[dimensions['512']][key] for key in strategies], width, label='Hidden Dim = 512')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Graph Reordering Strategy')
ax.set_ylabel('Speedup (relative to average of random order graphs)')
ax.set_xticks(x)
ax.set_xticklabels(strategies, rotation=45, ha="right")

# Set y-axis limits from 0 and ticks at intervals of 0.5
ax.set_ylim(0, 1.2)
ax.set_yticks(np.arange(0, 1.2, 1))

# Move the legend outside of the plot
ax.legend(loc='best', bbox_to_anchor=(1, 1))

## Adjust layout
fig.tight_layout()

# Save the figure
fig.savefig('speedup_exp1_hidden_dimension_dgl.png', dpi=300, bbox_inches='tight')

# Display the plot (optional, comment out if running on a server without display)
plt.show()

