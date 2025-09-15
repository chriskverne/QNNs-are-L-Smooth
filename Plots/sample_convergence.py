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



# fix,ax = plt.subplots(figsize=(10, 6))
# ax.set_facecolor('gainsboro')
# # Main plot
# plt.plot(n_samples, max_hessian_norms_1q, 'o-', linewidth=5, markersize=10,
#          color='blue', markerfacecolor='darkblue', markeredgecolor='white',
#          markeredgewidth=1.5, label='Max Hessian Norm')
# plt.plot(n_samples, max_hessian_norms_2q, 'o-', linewidth=5, markersize=10,
#          color='blue', markerfacecolor='darkblue', markeredgecolor='white',
#          markeredgewidth=1.5, label='Max Hessian Norm')
# plt.plot(n_samples, max_hessian_norms_4q, 'o-', linewidth=5, markersize=10,
#          color='blue', markerfacecolor='darkblue', markeredgecolor='white',
#          markeredgewidth=1.5, label='Max Hessian Norm')
#
# # Add horizontal line showing convergence reference (mean of last few points)
# convergence_value = np.mean(max_hessian_norms_2q[-3:])  # Average of last 3 points
# plt.axhline(y=convergence_value, color='red', linestyle='--', alpha=0.7,
#             label=f'Convergence Est. ({convergence_value:.3f})')
#
# # Add confidence band around convergence value (optional)
# std_last_few = np.std(max_hessian_norms_2q[:])
# plt.fill_between([min(n_samples), max(n_samples)],
#                 convergence_value - std_last_few,
#                 convergence_value + std_last_few,
#                 alpha=0.2, color='red', label=f'±{np.std(max_hessian_norms_2q[:]):.4f} std')
#
# # Formatting
# plt.xlabel('Number of Random Parameter Samples', fontsize=14)
# plt.ylabel('Largest Observed Hessian Norm', fontsize=14)
# ax.minorticks_on()
# ax.grid(True, which='major', linestyle='-', linewidth=2.0, alpha=1)
# ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=0.4)
# plt.legend(fontsize=12,fancybox=True,
#               edgecolor='black',
#               facecolor='white',
#               framealpha=1.0)
#
# # Add text annotation for sample size recommendation
# knee_point_idx = -1  # You can adjust this based on where you see convergence
# for i in range(1, len(n_samples)-1):
#     if abs(max_hessian_norms_2q[i+1] - max_hessian_norms_2q[i]) < 0.05:  # Adjust threshold
#         knee_point_idx = i
#         break
#
# # plt.ylim(0, max(max_hessian_norms) + 0.1)
# ax.tick_params(labelsize=12)
#
# plt.tight_layout()
# plt.show()
#
# # # Optional: Print convergence statistics
# # print(f"Convergence Statistics:")
# # print(f"All values: {max_hessian_norms[:]}")
# # print(f"Mean of all: {np.mean(max_hessian_norms[:]):.4f}")
# # print(f"Std of all {np.std(max_hessian_norms[:]):.4f}")
# # print(f"Coefficient of variation: {np.std(max_hessian_norms[-3:]) / np.mean(max_hessian_norms[-3:]) * 100:.2f}%")
