import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# Define the base directory and experiment sets
base_dir = '../experiments/Experiment 1 - Hidden Dimensions'
sets = {
    '16': 'Set 1 - Hidden Dimensions = 16/pyg',
    '64': 'Set 2 - Hidden Dimensions = 64/pyg',
    '512': 'Set 3 - Hidden Dimensions = 512/pyg'
}

# Function to extract data from file
def extract_data(file_path):
    strategy = None
    cache_misses = None

    # Regular expressions to find the reordering strategy and cache misses
    cache_misses_pattern = re.compile(r'cache-misses\s+#\s+(\d+\.\d+)')
    strategy_pattern = re.compile(r"'reordering_strategy':\s*'([^']+)'")

    with open(file_path, 'r') as file:
        contents = file.readlines()

        for line in contents:
            # Search for reordering strategy using regex
            if strategy is None:
                strategy_match = strategy_pattern.search(line)
                if strategy_match:
                    strategy = strategy_match.group(1)

            # Search for cache misses in lines matching the pattern
            if cache_misses is None:
                cache_misses_match = cache_misses_pattern.search(line)
                if cache_misses_match:
                    cache_misses = float(cache_misses_match.group(1))

            # If both values are found, stop further processing
            if strategy is not None and cache_misses is not None:
                break

    return strategy, cache_misses

# Initialize results dictionary
results = {
    "16": {"GCN": {}, "GRAPHSAGE": {}},
    "64": {"GCN": {}, "GRAPHSAGE": {}},
    "512": {"GCN": {}, "GRAPHSAGE": {}}
}

# Extract data from each set
for dimension, folder in sets.items():
    set_dir = os.path.join(base_dir, folder)

    for file_name in os.listdir(set_dir):
        # Only process files named "1.counts" to "34.counts"
        if not file_name.endswith('.counts') or not file_name.split('.')[0].isdigit():
            continue

        # Determine the model type based on file name
        model = "GCN" if "GCN" in file_name else "GRAPHSAGE"
        file_path = os.path.join(set_dir, file_name)

        # Extract data from file
        strategy, cache_misses = extract_data(file_path)

        # Store data in the results dictionary if both values are found
        if strategy is not None and cache_misses is not None:
            if strategy not in results[dimension][model]:
                results[dimension][model][strategy] = []
            results[dimension][model][strategy].append(cache_misses)

# Initialize averaged results dictionary
averaged_results = {
    "16": {},
    "64": {},
    "512": {}
}

# Compute the average cache misses for the same reordering strategy
for dimension in results:
    for strategy in set(results[dimension]['GCN'].keys()).union(results[dimension]['GRAPHSAGE'].keys()):
        # Get cache misses for GCN and GRAPHSAGE for the current strategy
        gcn_misses = results[dimension]['GCN'].get(strategy, [])
        graphsage_misses = results[dimension]['GRAPHSAGE'].get(strategy, [])

        # Combine all cache misses for averaging
        all_misses = gcn_misses + graphsage_misses

        if all_misses:
            # Calculate the average
            average_miss = sum(all_misses) / len(all_misses)
            averaged_results[dimension][strategy] = average_miss

print("Results:", json.dumps(results, indent=4))
print("Averaged Results:", json.dumps(averaged_results, indent=4))

# Extract the reordering strategies and dimensions
strategies = [s for s in averaged_results["16"].keys() if s not in ["rand-0", "rand-1", "rand-2"]]
dimensions = list(averaged_results.keys())

# Prepare data for plotting
data = {dim: [averaged_results[dim][strategy] for strategy in strategies] for dim in dimensions}

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Set the width of each bar and the positions of the bars on the x-axis
width = 0.25
x = np.arange(len(strategies))

# Create the grouped bars
for i, dim in enumerate(dimensions):
    ax.bar(x + i * width, data[dim], width, label=f'Hidden Dim = {dim}')

# Customize the plot
ax.set_ylabel('Cache Misses in %')
ax.set_xlabel('Reordering Strategies')
ax.set_xticks(x + width)
ax.set_xticklabels(strategies, rotation=45, ha='right')

# Set y-axis limits from 0 and ticks at intervals of 0.5
ax.set_ylim(0, 60)
ax.set_yticks(np.arange(0, 60, 10))

# Add a legend
ax.legend(loc='best', bbox_to_anchor=(1, 1))

# Adjust layout to ensure everything fits
plt.tight_layout()

# Save the figure
plt.savefig('cache_misses_exp1_pyg.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
