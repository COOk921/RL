import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Define reasonable weight ranges (in tons)
WEIGHT_RANGE_20FT = (3, 30)
WEIGHT_RANGE_40FT = (3, 30)

def map_weight_to_level(weight):
    """Maps a weight in the range [2, 30] to a level from 1 to 10."""
    if 2 <= weight <= 3:
        return 1
    elif 3 < weight <= 6:
        return 2
    elif 6 < weight <= 9:
        return 3
    elif 9 < weight <= 12:
        return 4
    elif 12 < weight <= 15:
        return 5
    elif 15 < weight <= 18:
        return 6
    elif 18 < weight <= 21:
        return 7
    elif 21 < weight <= 24:
        return 8
    elif 24 < weight <= 27:
        return 9
    elif 27 < weight <= 30:
        return 10
    else:
        # Handle weights slightly outside due to rounding, or errors
        if weight < 2:
            print(f"Warning: Weight {weight} is below range [2, 30], mapping to level 1.")
            return 1
        elif weight > 30:
            print(f"Warning: Weight {weight} is above range [2, 30], mapping to level 10.")
            return 10
        else:
            # This case should ideally not be reached if generation is correct
            print(f"Error: Weight {weight} is out of expected bounds and mapping logic.")
            return None # Or raise an error


# --- Weight Generation Functions ---

def generate_mixed_weights(n, min_w, max_w):
    """Generates weights with a bimodal (light/heavy mix) distribution."""
    weights = np.zeros(n)
    # Define approximate ranges for 'light' and 'heavy' peaks
    mid_point = min_w + (max_w - min_w) / 3 # Adjust split point if needed
    heavy_start = min_w + 2 * (max_w - min_w) / 3

    light_peak_range = (min_w, mid_point)
    heavy_peak_range = (heavy_start, max_w)

    # Roughly half light, half heavy
    num_light = n // 2
    num_heavy = n - num_light

    weights[:num_light] = np.random.uniform(light_peak_range[0], light_peak_range[1], size=num_light)
    weights[num_light:] = np.random.uniform(heavy_peak_range[0], heavy_peak_range[1], size=num_heavy)

    np.random.shuffle(weights) # Mix them up
    return np.round(weights, 2) # Round to 2 decimal places

def generate_heavy_biased_weights(n, min_w, max_w, alpha=5, beta=2):
    """Generates weights biased towards the maximum value using Beta distribution."""
    # Beta(alpha, beta) with alpha > beta skews towards 1
    # Scale the [0, 1] output of beta distribution to [min_w, max_w]
    weights_scaled_01 = np.random.beta(alpha, beta, size=n)
    weights = min_w + weights_scaled_01 * (max_w - min_w)
    return np.round(weights, 2)

def generate_light_biased_weights(n, min_w, max_w, alpha=2, beta=5):
    """Generates weights biased towards the minimum value using Beta distribution."""
    # Beta(alpha, beta) with beta > alpha skews towards 0
    # Scale the [0, 1] output of beta distribution to [min_w, max_w]
    weights_scaled_01 = np.random.beta(alpha, beta, size=n)
    weights = min_w + weights_scaled_01 * (max_w - min_w)
    return np.round(weights, 2)

def generate_container_weights(num_containers, container_type, distribution='mixed'):
    """Main function to generate weights based on type and distribution."""
    if container_type == '20ft':
        min_w, max_w = WEIGHT_RANGE_20FT
    elif container_type == '40ft':
        min_w, max_w = WEIGHT_RANGE_40FT
    else:
        raise ValueError("container_type must be '20ft' or '40ft'")

    if distribution == 'mixed':
        return generate_mixed_weights(num_containers, min_w, max_w)
    elif distribution == 'heavy':
        # You can adjust alpha and beta here to control the skewness
        return generate_heavy_biased_weights(num_containers, min_w, max_w, alpha=5, beta=1.5)
    elif distribution == 'light':
        # You can adjust alpha and beta here
        return generate_light_biased_weights(num_containers, min_w, max_w, alpha=1.5, beta=5)
    elif distribution == 'uniform':
        # Added uniform for comparison baseline if needed
         weights = np.random.uniform(min_w, max_w, size=num_containers)
         return np.round(weights, 2)
    else:
        raise ValueError("distribution must be 'mixed', 'heavy', 'light', or 'uniform'")

# --- Example Usage & Visualization ---
num_samples = 100 # Generate 1000 weights for visualization

weights_mixed_20ft = generate_container_weights(num_samples, '20ft', 'mixed')
weights_heavy_20ft = generate_container_weights(num_samples, '20ft', 'heavy')
weights_light_20ft = generate_container_weights(num_samples, '20ft', 'light')
weights_uniform_20ft = generate_container_weights(num_samples, '20ft', 'uniform')

# Plotting histograms to verify
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle('Simulated 20ft Container Weight Distributions (Tons)')

axs[0, 0].hist(weights_mixed_20ft, bins=20, color='skyblue', edgecolor='black')
axs[0, 0].set_title('混合分布')
# axs[0, 0].set_xlabel('重量 (t)')
axs[0, 0].set_ylabel('频率')

axs[0, 1].hist(weights_heavy_20ft, bins=20, color='salmon', edgecolor='black')
axs[0, 1].set_title('偏重分布')
# axs[0, 1].set_xlabel('重量 (t)')
# axs[0, 1].set_ylabel('频率')

axs[1, 0].hist(weights_light_20ft, bins=20, color='lightgreen', edgecolor='black')
axs[1, 0].set_title('偏轻分布')
axs[1, 0].set_xlabel('重量 (t)')
axs[1, 0].set_ylabel('频率')

axs[1, 1].hist(weights_uniform_20ft, bins=20, color='gold', edgecolor='black')
axs[1, 1].set_title('均匀分布')
axs[1, 1].set_xlabel('重量 (t)')
# axs[1, 1].set_ylabel('频率')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
# plt.savefig('container_weight_distributions.png', dpi=500, bbox_inches='tight')
# plt.show()

# --- How to use in your dataset generation ---
# Let's say you are generating data for one specific stowage problem instance
num_20ft_containers = 50

# Decide which distribution this instance should follow, e.g., 'heavy'
distribution_for_this_instance = 'light'  # or 'light', 'mixed', uniform,

weights_20ft = generate_container_weights(num_20ft_containers, '20ft', distribution_for_this_instance)

print("Mapping weights to levels...")
vectorized_map_weight_to_level = np.vectorize(map_weight_to_level)
weight_levels = vectorized_map_weight_to_level(weights_20ft)


# Check if any None values were returned by mapping (indicates error)
output_filename = f'./ContainerData/container_levels_{num_20ft_containers}_{distribution_for_this_instance}.txt'


if np.any(weight_levels == None):
     print("Error encountered during weight mapping. Please check warnings.")
else:
    # Ensure levels are integers
    weight_levels = weight_levels.astype(int)

    # 4. Write to File
    print(f"Writing levels to file: {output_filename}")
    try:
        with open(output_filename, 'w') as f:
            for level in weight_levels:
                f.write(f"{level} ")
        print("File written successfully.")
    except Exception as e:
        print(f"Error writing to file: {e}")