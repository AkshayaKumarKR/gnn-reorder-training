import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Base directory and settings for the experiments
base_dir = '../experiments/Experiment 3 - Layers'

sets = {
    'Set 1 - Layers = 2': 'pyg',
    'Set 2 - Layers = 3': 'pyg',
    'Set 3 - Layers = 4': 'pyg'
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


print(results)
print("Combined Averaged Results:", combined_averaged_results)

# Prepare the data for plotting
layers = {'2': 'Set 1 - Layers = 2', '3': 'Set 2 - Layers = 3', '4': 'Set 3 - Layers = 4'}
strategies = sorted(list(combined_averaged_results[layers['2']].keys()))

x = np.arange(len(strategies))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Plotting the bars for each number of layers
rects1 = ax.bar(x - width, [combined_averaged_results[layers['2']][key] for key in strategies], width, label='Number of Layers = 2')
rects2 = ax.bar(x, [combined_averaged_results[layers['3']][key] for key in strategies], width, label='Number of Layers = 3')
rects3 = ax.bar(x + width, [combined_averaged_results[layers['4']][key] for key in strategies], width, label='Number of Layers = 4')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Graph Reordering Strategy')
ax.set_ylabel('Average Training Time (seconds)')
ax.set_title('Average Training Time by Graph Reordering Strategy and Number of Layers')
ax.set_xticks(x)
ax.set_xticklabels(strategies, rotation=45, ha="right")

# Move the legend outside of the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

## Adjust layout
fig.tight_layout()

# Save the figure
fig.savefig('training_time_comparison_exp3_number_of_layers_pyg.png', dpi=300, bbox_inches='tight')

# Display the plot (optional, comment out if running on a server without display)
# plt.show()

# Close the plot to free up memory
plt.close(fig)
