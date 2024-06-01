import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Example data
groups = ['Group 1', 'Group 2', 'Group 3']
categories = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5', 'Cat 6']

# Accuracy values for each group and category
accuracies = np.array([
    [0.85, 0.88, 0.90, 0.87, 0.86, 0.89],  # Group 1
    [0.80, 0.82, 0.84, 0.81, 0.83, 0.85],  # Group 2
    [0.78, 0.79, 0.82, 0.80, 0.77, 0.81]   # Group 3
])

# Parameters
n_groups = len(groups)
n_categories = len(categories)
bar_width = 0.25  # Width of each bar
indices = np.arange(n_categories)

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars for each group
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each group
for i in range(n_groups):
    bar_positions = indices + i * bar_width
    bars = ax.bar(bar_positions, accuracies[i], bar_width, label=groups[i], color=colors[i])

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add grid lines
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Performance of Model Groups Across Categories')
ax.set_xticks(indices + bar_width * (n_groups - 1) / 2)
ax.set_xticklabels(categories)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()