import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_results():
    # Load data
    data = pd.read_csv('mean_results.csv')
    
    # Color mapping for memory modes
    color_map = {0: 'blue', 1: 'green', 2: 'red'}
    
    # Unique maps
    maps = data['Map'].unique()
    
    # Create a plot for each Map
    for map_id in maps:
        # Create a wider figure. Increase the figure width here.
        fig, ax = plt.subplots(figsize=(12, 6))  # Increased the width to 12 inches for better visibility
    
        # Filter data for the current Map
        map_data = data[data['Map'] == map_id]
    
        # Get the unique swarm sizes for ordering on the x-axis
        unique_sizes = sorted(map_data['Swarm Size'].unique())
    
        # Define the width of each bar and the additional spacing between bars within a group
        bar_width = 0.25
        intra_group_spacing = 0.05  # Small gap between bars within the same group
    
        # Calculate the total width of all swarm size groups within one memory mode group
        mode_group_width = (len(unique_sizes) * bar_width) + ((len(unique_sizes) - 1) * intra_group_spacing)
    
        # Create labels list and positions for ticks
        x_labels = []
        all_positions = []
    
        # Plot bars grouped by memory mode
        for mode, color in color_map.items():
            mode_data = map_data[map_data['Memory Mode'] == mode]
            mode_label = f'Memory Mode {mode}'
            for i, size in enumerate(unique_sizes):
                # Calculate position of each bar within the group
                x_pos = i * (bar_width + intra_group_spacing) + mode * (mode_group_width + 0.1)  # Adding space between groups of memory modes
                mean_time = mode_data[mode_data['Swarm Size'] == size]['Mean Time Taken'].values
                if mean_time.size > 0:
                    ax.bar(x_pos, mean_time[0], color=color, width=bar_width, label=mode_label if i == 0 else "")
                # Add label and adjust position for ticks
                x_labels.append(f'{size}')
                all_positions.append(x_pos)
    
        # Customizing the plot
        ax.set_xlabel('Swarm Size')
        ax.set_ylabel('Mean Time Taken')
        ax.set_title(f'Mean Time Taken by Swarm Size for Map {map_id}')
    
        # Set x-ticks and x-tick labels
        ax.set_xticks(all_positions)
        ax.set_xticklabels(x_labels, rotation=45)  # Each position gets a swarm size label
    
        # Add legend outside the plot to the right
        ax.legend(title="Memory Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.tight_layout()
        plt.show()
    
def plot_cv_bar_chart():
    # Load the coefficient of variance data
    data = pd.read_csv('coefficient_variance_results.csv')

    # Prepare the plot
    plt.figure(figsize=(12, 6))  # Increased figure width for better visualization

    # Set color map and details for plotting
    color_map = {0: 'blue', 1: 'green', 2: 'red'}
    memory_modes = data['Memory Mode'].unique()
    unique_sizes = sorted(data['Swarm Size'].unique())

    # Define the width of each bar and the additional spacing between bars within a group
    bar_width = 0.25
    intra_group_spacing = 0.05  # Small gap between bars within the same group

    # Calculate the total width of all swarm size groups within one memory mode group
    mode_group_width = (len(unique_sizes) * bar_width) + ((len(unique_sizes) - 1) * intra_group_spacing)

    # Create labels list and positions for ticks
    x_labels = []
    all_positions = []

    # Create bars for each memory mode
    for mode, color in color_map.items():
        mode_data = data[data['Memory Mode'] == mode]
        mode_label = f'Memory Mode {mode}'
        for i, size in enumerate(unique_sizes):
            # Calculate position of each bar within the group
            x_pos = i * (bar_width + intra_group_spacing) + mode * (mode_group_width + 0.1)  # Adding space between groups of memory modes
            cv_value = mode_data[mode_data['Swarm Size'] == size]['Coefficient of Variance'].values
            if cv_value.size > 0:
                plt.bar(x_pos, cv_value[0], color=color, width=bar_width, label=mode_label if i == 0 else "")
            # Add label and adjust position for ticks
            x_labels.append(f'{size}')
            all_positions.append(x_pos)

    # Adding labels and title
    plt.xlabel('Swarm Size')
    plt.ylabel('Coefficient of Variance')
    plt.title('Coefficient of Variance of Results from All Maps')

    # Set x-ticks and x-tick labels
    plt.xticks(all_positions, x_labels, rotation=45)

    # Adding legend outside the plot to the right
    plt.legend(title="Memory Mode", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    plt.tight_layout()
    plt.show()

# Call the function to plot the data
plot_mean_results()
plot_cv_bar_chart()