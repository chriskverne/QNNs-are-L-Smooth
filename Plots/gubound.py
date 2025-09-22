import matplotlib.pyplot as plt
import numpy as np

data = {
    2: [("Ising Model", 3.7272, 16.9706, 24.0000),("Heisenberg Model", 2.5893, 18.0000, 18.0000),("Mixed-Field Model", 4.0398, 28.3246, 38.4000)],
    4: [("Ising Model", 2.9703, 33.9411, 48.0000), ("Heisenberg Model", 1.9727, 36.0000, 36.0000),("Mixed-Field Model", 3.7227, 56.6493, 76.8000)],
    8: [("Ising Model", 3.0685, 67.8823, 96.0000),("Heisenberg Model", 2.1658, 72.0000, 72.0000),("Mixed-Field Model", 3.6767, 113.2985, 153.6000)]
}

# --- 2. Plotting Configuration ---
model_order = ["Heisenberg Model", "Ising Model", "Mixed-Field Model"]
n_values = sorted(data.keys()) # [2, 4, 8]
bar_width = 0.20        # Set the desired bar width
group_spacing_factor = 1.5 # Increase this for more space between model groups

# Prepare data in a structured way for easier plotting
model_data = {model: {} for model in model_order}
for n, vals in data.items():
    for model_name, lmax, our_bound, gu_bound in vals:
        if model_name in model_data:
            # Pre-calculate the ratios and store them
            model_data[model_name][n] = (lmax / our_bound * 100, lmax / gu_bound * 100)

# Define consistent colors and hatches
colors = {
    2: ('#4c72b0', '#82a3d1'),  # Dark/Light Blue for n=2
    4: ('#55a868', '#82c092'),  # Dark/Light Green for n=4
    8: ('#8172b2', '#ad9ed1')   # Dark/Light Purple for n=8
}
hatches = ('///', 'ooo') # Our, Gu


# --- 3. Plot Generation ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_facecolor('gainsboro')
ax.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.8, zorder=0)

# Calculate the base positions for each model group with spacing
x_base = np.arange(len(model_order)) * group_spacing_factor
num_bar_pairs = len(n_values)
total_group_width = num_bar_pairs * 2 * bar_width

# Loop through each model to plot its group of bars
for i, model_name in enumerate(model_order):
    # Calculate the starting position to center the whole group of bars on its tick
    group_start_pos = x_base[i] - total_group_width / 2 + bar_width / 2

    # Loop through each system size (n) to plot a pair of bars
    for j, n in enumerate(n_values):
        our_ratio, gu_ratio = model_data[model_name][n]
        our_color, gu_color = colors[n]
        bar_pair_offset = j * 2 * bar_width

        # Plot "Our Bound" bar
        ax.bar(group_start_pos + bar_pair_offset, our_ratio, bar_width,
               color=our_color, hatch=hatches[0], edgecolor='black', zorder=3,
               label=f"$L_{{max}}$ / our bound (n={n})" if i == 0 else "")

        # Plot "Gu Bound" bar
        ax.bar(group_start_pos + bar_pair_offset + bar_width, gu_ratio, bar_width,
               color=gu_color, hatch=hatches[1], edgecolor='black', zorder=3,
               label=f"$L_{{max}}$ / Gu bound (n={n})" if i == 0 else "")


# --- 4. Final Customization ---
ax.set_ylabel(r'$L_{\mathrm{max}} \,/\, L_{\mathrm{upper}}$ (%)', fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, 25)

# Set x-ticks and labels to be in the center of each group
ax.set_xticks(x_base)
ax.set_xticklabels(model_order, fontsize=16)
ax.tick_params(axis='x', length=0) # Hide tick marks on x-axis
ax.set_xlabel('')

# Create the two-row horizontal legend above the plot
handles, labels = ax.get_legend_handles_labels()
reordered_handles = [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
reordered_labels = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]]
fig.legend(reordered_handles, reordered_labels,
           ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           frameon=False, fontsize='large')

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for legend
# plt.show()


data = {
    2: [("Ising Model", 3.7272, 16.9706, 24.0000),
        ("Heisenberg Model", 2.5893, 18.0000, 18.0000),
        ("Mixed-Field Model", 4.0398, 28.3246, 38.4000)],
    4: [("Ising Model", 2.9703, 33.9411, 48.0000),
        ("Heisenberg Model", 1.9727, 36.0000, 36.0000),
        ("Mixed-Field Model", 3.7227, 56.6493, 76.8000)],
    8: [("Ising Model", 3.0685, 67.8823, 96.0000),
        ("Heisenberg Model", 2.1658, 72.0000, 72.0000),
        ("Mixed-Field Model", 3.6767, 113.2985, 153.6000)]
}


def compare_ratios(data):
    for n_qubits, entries in data.items():
        print(f"\n=== {n_qubits} qubits ===")
        for obs, Lmax, B1, B2 in entries:
            ratio1 = Lmax / B1 if B1 != 0 else float('inf')
            ratio2 = Lmax / B2 if B2 != 0 else float('inf')

            if ratio1 == ratio2:
                status = "same"
            elif ratio1 < ratio2:
                diff = (1 - ratio1 / ratio2) * 100
                status = f"{diff:.1f}% tighter"
            else:
                diff = (ratio1 / ratio2 - 1) * 100
                status = f"{diff:.1f}% higher"

            print(f"{obs:20}: Lmax/Bound1={ratio1:.4f}, Lmax/Bound2={ratio2:.4f} → {status}")


compare_ratios(data)

# # 2q 2l
# # --- Running experiment for: Ising Model ---
# # P = 12
# # Sum |c_k|: 2.0000, ||M||_2: 1.4142
# # Your Bound: 16.9706, Gu Bound: 24.0000
# # Empirical L_max: 3.7272
# #
# # --- Running experiment for: Heisenberg Model ---
# # P = 12
# # Sum |c_k|: 1.5000, ||M||_2: 1.5000
# # Your Bound: 18.0000, Gu Bound: 18.0000
# # Empirical L_max: 2.5893
# #
# # --- Running experiment for: Mixed-Field Model ---
# # P = 12
# # Sum |c_k|: 3.2000, ||M||_2: 2.3604
# # Your Bound: 28.3246, Gu Bound: 38.4000
# # Empirical L_max: 4.0398
# #
# # 4q 2l
# # --- Running experiment for: Ising Model ---
# # P = 24
# # Sum |c_k|: 2.0000, ||M||_2: 1.4142
# # Your Bound: 33.9411, Gu Bound: 48.0000
# # Empirical L_max: 2.9703
# #
# # --- Running experiment for: Heisenberg Model ---
# # P = 24
# # Sum |c_k|: 1.5000, ||M||_2: 1.5000
# # Your Bound: 36.0000, Gu Bound: 36.0000
# # Empirical L_max: 1.9727
# #
# # --- Running experiment for: Mixed-Field Model ---
# # P = 24
# # Sum |c_k|: 3.2000, ||M||_2: 2.3604
# # Your Bound: 56.6493, Gu Bound: 76.8000
# # Empirical L_max: 3.7227
# #
# # 8q 2l
# # --- Running experiment for: Ising Model ---
# # P = 48
# # Sum |c_k|: 2.0000, ||M||_2: 1.4142
# # Your Bound: 67.8823, Gu Bound: 96.0000
# # Empirical L_max: 3.0685
# # --- Running experiment for: Heisenberg Model ---
# # P = 48
# # Sum |c_k|: 1.5000, ||M||_2: 1.5000
# # Your Bound: 72.0000, Gu Bound: 72.0000
# # Empirical L_max: 2.1658
# #
# # --- Running experiment for: Mixed-Field Model ---
# # P = 48
# # Sum |c_k|: 3.2000, ||M||_2: 2.3604
# # Your Bound: 113.2985, Gu Bound: 153.6000
# # Empirical L_max: 3.6767
#
#
#
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # --- 1. Data Definition ---
# # The data provided, structured in a dictionary.
# # Correctly interpreted format: {num_qubits: [(Model, L_max, Ours_Bound, Gu_Bound), ...]}
# data = {
#     2: [("Ising Model", 3.7272, 16.9706, 24.0000),("Heisenberg Model", 2.5893, 18.0000, 18.0000),("Mixed-Field Model", 4.0398, 28.3246, 38.4000)],
#     4: [("Ising Model", 2.9703, 33.9411, 48.0000), ("Heisenberg Model", 1.9727, 36.0000, 36.0000),("Mixed-Field Model", 3.7227, 56.6493, 76.8000)],
#     8: [("Ising Model", 3.0685, 67.8823, 96.0000),("Heisenberg Model", 2.1658, 72.0000, 72.0000),("Mixed-Field Model", 3.6767, 113.2985, 153.6000)]
# }
#
# # --- 2. Data Processing (Corrected) ---
# plot_data = []
# # *** The key correction is here, where the tuple is unpacked correctly ***
# for n_qubits, models in data.items():
#     for model_name, l_max, ours_bound, gu_bound in models:
#         # Ratio for the Gu et al. bound
#         plot_data.append({
#             'Model': model_name,
#             'n_qubits': n_qubits,
#             'Bound Type': 'Gu Bound',
#             'Ratio (%)': (l_max / gu_bound) * 100
#         })
#         # Ratio for this paper's bound
#         plot_data.append({
#             'Model': model_name,
#             'n_qubits': n_qubits,
#             'Bound Type': 'Our Bound',
#             'Ratio (%)': (l_max / ours_bound) * 100
#         })
#
# df = pd.DataFrame(plot_data)
#
# # --- 3. Plot Configuration ---
# # sns.set_theme(style="whitegrid")
# sns.set_context("notebook", font_scale=1.1)
#
# # Define the order for the models on the x-axis
# model_order = ["Heisenberg Model", "Ising Model", "Mixed-Field Model"]
# df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
# df = df.sort_values('Model')
#
# # Create a combined column for the legend and define the sorting order
# df['Legend'] = df.apply(lambda row: f"{row['Bound Type']} (n={row['n_qubits']})", axis=1)
# hue_order = [
#     'Gu Bound (n=2)', 'Our Bound (n=2)',
#     'Gu Bound (n=4)', 'Our Bound (n=4)',
#     'Gu Bound (n=8)', 'Our Bound (n=8)'
# ]
#
# # Define a color palette that groups by system size (n)
# # Lighter shade for Gu Bound, darker for Our Bound
# color_palette = {
#     'Gu Bound (n=2)': '#82a3d1', 'Our Bound (n=2)': '#4c72b0',  # Blues
#     'Gu Bound (n=4)': '#82c092', 'Our Bound (n=4)': '#55a868',  # Greens
#     'Gu Bound (n=8)': '#ad9ed1', 'Our Bound (n=8)': '#8172b2'   # Purples
# }
#
# # --- 4. Generate the Plot ---
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_facecolor('gainsboro')
#
# sns.barplot(
#     data=df,
#     x='Model',
#     y='Ratio (%)',
#     hue='Legend',
#     hue_order=hue_order, # Apply the sorting order
#     palette=color_palette,
#     ax=ax
# )
#
# # --- 5. Customization and Labels ---
# ax.tick_params(labelsize=16)
# ax.set_ylabel(r'$L_{\mathrm{max}} \,/\, L_{\mathrm{upper}}$ (%)', fontsize=20)
# ax.grid(True, which='major', linestyle='-', linewidth=2.0, alpha=1)
# ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=0.4)
# ax.legend(title='Bound / System Size')
# ax.set_ylim(0, 25) # Set a fixed y-limit for better comparison
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # --- 1. Data Definition ---
# # The data provided, structured in a dictionary.
# data = {
#     2: [("Ising Model", 3.7272, 16.9706, 24.0000),("Heisenberg Model", 2.5893, 18.0000, 18.0000),("Mixed-Field Model", 4.0398, 28.3246, 38.4000)],
#     4: [("Ising Model", 2.9703, 33.9411, 48.0000), ("Heisenberg Model", 1.9727, 36.0000, 36.0000),("Mixed-Field Model", 3.7227, 56.6493, 76.8000)],
#     8: [("Ising Model", 3.0685, 67.8823, 96.0000),("Heisenberg Model", 2.1658, 72.0000, 72.0000),("Mixed-Field Model", 3.6767, 113.2985, 153.6000)]
# }
#
# # --- 2. Data Processing ---
# plot_data = []
# for n_qubits, models in data.items():
#     for model_name, l_max, ours_bound, gu_bound in models:
#         # Ratio for the Gu et al. bound
#         plot_data.append({
#             'Model': model_name,
#             'n_qubits': n_qubits,
#             'Bound Type': 'Gu Bound',
#             'Ratio (%)': (l_max / gu_bound) * 100
#         })
#         # Ratio for this paper's bound
#         plot_data.append({
#             'Model': model_name,
#             'n_qubits': n_qubits,
#             'Bound Type': 'Our Bound',
#             'Ratio (%)': (l_max / ours_bound) * 100
#         })
#
# df = pd.DataFrame(plot_data)
#
# # --- 3. Plot Configuration ---
# sns.set_context("notebook", font_scale=1.1)
#
# # Define the order for the models on the x-axis
# model_order = ["Heisenberg Model", "Ising Model", "Mixed-Field Model"]
# df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
# df = df.sort_values('Model')
#
# # Create a combined column for the legend and define the sorting order
# df['Legend'] = df.apply(lambda row: f"{row['Bound Type']} (n={row['n_qubits']})", axis=1)
# hue_order = [
#     'Gu Bound (n=2)', 'Our Bound (n=2)',
#     'Gu Bound (n=4)', 'Our Bound (n=4)',
#     'Gu Bound (n=8)', 'Our Bound (n=8)'
# ]
#
# # Define a color palette that groups by system size (n)
# color_palette = {
#     'Gu Bound (n=2)': '#82a3d1', 'Our Bound (n=2)': '#4c72b0',  # Blues
#     'Gu Bound (n=4)': '#82c092', 'Our Bound (n=4)': '#55a868',  # Greens
#     'Gu Bound (n=8)': '#ad9ed1', 'Our Bound (n=8)': '#8172b2'   # Purples
# }
#
# # --- 4. Generate the Plot ---
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_facecolor('gainsboro')
#
# sns.barplot(
#     data=df,
#     x='Model',
#     y='Ratio (%)',
#     hue='Legend',
#     hue_order=hue_order,
#     palette=color_palette,
#     ax=ax
# )
#
# # --- 5. Customization and Labels ---
# ax.set_xlabel('')
#
# # --- Handle the legend styling ---
# handles, labels = ax.get_legend_handles_labels()
# ax.get_legend().remove()
#
# # Create the new labels with LaTeX formatting for L_max
# # This is the only line that has been changed
# new_labels = [fr'$L_{{\mathrm{{max}}}}$ / {l.replace("Bound", "bound")}' for l in labels]
#
# reordered_handles = [handles[1], handles[3], handles[5], handles[0], handles[2], handles[4]]
# reordered_labels = [new_labels[1], new_labels[3], new_labels[5], new_labels[0], new_labels[2], new_labels[4]]
#
# fig.legend(
#     reordered_handles,
#     reordered_labels,
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.02),
#     ncol=3,
#     frameon=False,
#     fontsize='medium'
# )
# # --- End of legend styling ---
#
# ax.tick_params(labelsize=16)
# ax.set_ylabel(r'$L_{\mathrm{max}} \,/\, L_{\mathrm{upper}}$ (%)', fontsize=20)
# ax.grid(True, which='major', linestyle='-', linewidth=2.0, alpha=1)
# ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=0.4)
# ax.set_ylim(0, 25)
#
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()