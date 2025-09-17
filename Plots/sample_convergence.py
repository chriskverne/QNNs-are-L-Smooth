import matplotlib.pyplot as plt
import numpy as np

# 1 qubit
# 10 samples:  4.825047114822578  50 samples:  5.268708754498552  100 samples:  5.4204775542482855  250 samples:  5.730351963044919  500 samples:  5.596185232254031  1000 samples:  5.834498554061603  2000 samples:  5.862191651651436 4000 samples:  5.8295817078117045
n_samples = [10, 50, 100, 250, 500, 1000, 2000, 4000]
max_hessian_norms_1q = [4.825047114822578, 5.268708754498552, 5.4204775542482855, 5.730351963044919, 5.596185232254031, 5.834498554061603, 5.862191651651436,  5.8295817078117045]

# 2 qubit
n_samples = [10, 50, 100, 250, 500, 1000, 2000, 4000]
max_hessian_norms_2q = [2.6805072639030025, 2.5269121135139216, 2.8551045049470862, 2.910669709448535, 2.9079474983639226, 2.96173548644182864, 2.968031440279984, 2.9363644775244446]

# 4 qubit
# 10 samples:  1.3769884137885868  50 samples:  1.3853800873983024  100 samples:  1.4430857608646108  250 samples:  1.423470577239144  500 samples:  1.460423354022187  1000 samples:  1.485758444124909
n_samples = [10, 50, 100, 250, 500, 1000, 2000, 4000]
max_hessian_norms_4q = [1.3769884137885868, 1.3853800873983024, 1.4430857608646108, 1.423470577239144, 1.460423354022187, 1.485758444124909, 1.47431454134964, 1.4900153244125391]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('gainsboro')

# Define distinct colors and styles for each line
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
markers = ['o', 's', '^']  # Circle, Square, Triangle
linestyles = ['-', '--', '-.']  # Solid, Dashed, Dash-dot
labels = ['n=1', 'n=2', 'n=4']
data_arrays = [max_hessian_norms_1q, max_hessian_norms_2q, max_hessian_norms_4q]

# Plot each line with distinct styling
for i, (data, color, marker, linestyle, label) in enumerate(zip(data_arrays, colors, markers, linestyles, labels)):
    plt.plot(n_samples, data, marker=marker, linestyle=linestyle,
             linewidth=4, markersize=8, color=color,
             markerfacecolor=color, markeredgecolor='white',
             markeredgewidth=1.5, label=label, alpha=0.9)

    # Add convergence reference line for each dataset
    convergence_value = np.mean(data[-3:])  # Average of last 3 points
    plt.axhline(y=convergence_value, color=color, linestyle=':', alpha=0.6, linewidth=2)

    # Add confidence band around convergence value
    std_data = np.std(data)
    plt.fill_between([min(n_samples), max(n_samples)],
                     convergence_value - std_data,
                     convergence_value + std_data,
                     alpha=0.4, color=color,
                     label=f'{label.split("(")[0].strip()} ±{std_data:.4f} std')

# Formatting
plt.xlabel('Number of Random Parameter Samples', fontsize=14)
plt.ylabel('Largest Observed Hessian Norm', fontsize=14)


# Enhanced grid
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-', linewidth=2, alpha=1)
ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=1)

# Enhanced legend with two columns to accommodate more entries
plt.legend(fontsize=11, fancybox=True, shadow=True,
           edgecolor='black', facecolor='white',
           framealpha=0.95, ncol=2, loc='center right')

# Improve tick formatting
ax.tick_params(labelsize=12, which='major', width=1.5)
ax.tick_params(labelsize=12, which='minor', width=1)

# Add subtle border around plot
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('darkgray')

plt.tight_layout()
plt.show()
