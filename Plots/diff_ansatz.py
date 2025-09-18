# 4 QUBIT, 1,2,4,5,8,10 LAYERS, 3 GATES, 100 S
# ==================== Running Architecture: Standard Ring (QNN 1) ====================
#   Testing 1 layers (P=12)...
#     -> L_max = 1.1836, L_upper = 12.0000, Ratio = 0.0986
#   Testing 2 layers (P=24)...
#     -> L_max = 1.3624, L_upper = 24.0000, Ratio = 0.0568
#   Testing 4 layers (P=48)...
#     -> L_max = 2.3714, L_upper = 48.0000, Ratio = 0.0494
#   Testing 6 layers (P=72)...
#     -> L_max = 3.2194, L_upper = 72.0000, Ratio = 0.0447
#   Testing 8 layers (P=96)...
#     -> L_max = 3.6541, L_upper = 96.0000, Ratio = 0.0381
#   Testing 10 layers (P=120)...
#     -> L_max = 3.7212, L_upper = 120.0000, Ratio = 0.0310

# ==================== Running Architecture: Linear CZ (QNN 2) ====================
#   Testing 1 layers (P=12)...
#     -> L_max = 0.2500, L_upper = 12.0000, Ratio = 0.0208
#   Testing 2 layers (P=24)...
#     -> L_max = 0.6790, L_upper = 24.0000, Ratio = 0.0283
#   Testing 4 layers (P=48)...
#     -> L_max = 1.5365, L_upper = 48.0000, Ratio = 0.0320
#   Testing 6 layers (P=72)...
#     -> L_max = 2.3548, L_upper = 72.0000, Ratio = 0.0327
#   Testing 8 layers (P=96)...
#     -> L_max = 3.5838, L_upper = 96.0000, Ratio = 0.0373
#   Testing 10 layers (P=120)...
#     -> L_max = 4.0662, L_upper = 120.0000, Ratio = 0.0339

# ==================== Running Architecture: All-to-All CNOT (QNN 3) ====================
#   Testing 1 layers (P=12)...
#     -> L_max = 0.4990, L_upper = 12.0000, Ratio = 0.0416
#   Testing 2 layers (P=24)...
#     -> L_max = 0.6896, L_upper = 24.0000, Ratio = 0.0287
#   Testing 4 layers (P=48)...
#     -> L_max = 1.7200, L_upper = 48.0000, Ratio = 0.0358
#   Testing 6 layers (P=72)...
#     -> L_max = 2.5634, L_upper = 72.0000, Ratio = 0.0356
#   Testing 8 layers (P=96)...
#     -> L_max = 3.8355, L_upper = 96.0000, Ratio = 0.0400
#   Testing 10 layers (P=120)...
#     -> L_max = 5.0711, L_upper = 120.0000, Ratio = 0.0423

# =======================================================================================
P_4q = [12, 24, 48, 72, 96, 120]
arch1_4q = [0.0986, 0.0568, 0.0494, 0.0447, 0.0381, 0.0310]
arch2_4q = [0.0208, 0.0283, 0.0320, 0.0327, 0.0373, 0.0339]
arch3_4q = [0.0416, 0.0287, 0.0358, 0.0356, 0.0400, 0.0423]

# 2 QUBIT, 1,5,10,15,20 LAYERS, 3 GATES, 100 S
P_2q = [6, 30, 60, 90, 120]
arch1_2q = [0.3197, 0.1603, 0.1570, 0.1320, 0.1390]
arch2_2q = [0.0833, 0.1450, 0.1358, 0.1467, 0.1418]
arch3_2q = [0.1663, 0.1443, 0.1671, 0.1667, 0.1858]
#
# ==================== Running Architecture: Standard Ring (QNN 1) ====================
#   Testing 1 layers (P=6)...
#     -> L_max = 1.9182, L_upper = 6, Ratio = 0.3197
#   Testing 5 layers (P=30)...
#     -> L_max = 4.8089, L_upper = 30, Ratio = 0.1603
#   Testing 10 layers (P=60)...
#     -> L_max = 9.4219, L_upper = 60, Ratio = 0.1570
#   Testing 15 layers (P=90)...
#     -> L_max = 11.8802, L_upper = 90, Ratio = 0.1320
#   Testing 20 layers (P=120)...
#     -> L_max = 16.7106, L_upper = 120, Ratio = 0.1390
#
# ==================== Running Architecture: Linear CZ (QNN 2) ====================
#   Testing 1 layers (P=6)...
#     -> L_max = 0.5000, L_upper = 6, Ratio = 0.0833
#   Testing 5 layers (P=30)...
#     -> L_max = 4.3498, L_upper = 30, Ratio = 0.1450
#   Testing 10 layers (P=60)...
#     -> L_max = 8.1466, L_upper = 60, Ratio = 0.1358
#   Testing 15 layers (P=90)...
#     -> L_max = 13.1993, L_upper = 90, Ratio = 0.1467
#   Testing 20 layers (P=120)...
#     -> L_max = 17.0156, L_upper = 120, Ratio = 0.1418
#
# ==================== Running Architecture: All-to-All CNOT (QNN 3) ====================
#   Testing 1 layers (P=6)...
#     -> L_max = 0.9981, L_upper = 6, Ratio = 0.1663
#   Testing 2 layers (P=30)...
#     -> L_max = 4.3279, L_upper = 30, Ratio = 0.1443
#   Testing 4 layers (P=60)...
#     -> L_max = 10.0279, L_upper = 60, Ratio = 0.1671
#   Testing 6 layers (P=90)...
#     -> L_max = 15.0040, L_upper =90, Ratio = 0.1667
#   Testing 8 layers (P=120)...
#     -> L_max = , L_upper = 120, Ratio = 0.1858


# import matplotlib.pyplot as plt
# P_4q = [12, 24, 48, 72, 96, 120]
# arch1_4q = [0.0986, 0.0568, 0.0494, 0.0447, 0.0381, 0.0310]
# arch2_4q = [0.0208, 0.0283, 0.0320, 0.0327, 0.0373, 0.0339]
# arch3_4q = [0.0416, 0.0287, 0.0358, 0.0356, 0.0400, 0.0423]
#
# # 2 QUBIT, 1,5,10,15,20 LAYERS, 3 GATES, 100 S
# P_2q = [6, 30, 60, 90, 120]
# arch1_2q = [0.3197, 0.1603, 0.1570, 0.1320, 0.1390]
# arch2_2q = [0.0833, 0.1450, 0.1358, 0.1467, 0.1418]
# arch3_2q = [0.1663, 0.1443, 0.1671, 0.1667, 0.1858]

















import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D # Import Line2D to create custom legend elements

# --- Data from your experiments ---

# 4-Qubit Data
P_4q = [12, 24, 48, 72, 96, 120]
# Ratios are multiplied by 100 to be plotted as percentages
arch1_4q = np.array([0.0986, 0.0568, 0.0494, 0.0447, 0.0381, 0.0410]) * 100
arch2_4q = np.array([0.0208, 0.0283, 0.0320, 0.0327, 0.0373, 0.0339]) * 100
arch3_4q = np.array([0.0416, 0.0287, 0.0358, 0.0356, 0.0400, 0.0423]) * 100

# 2-Qubit Data
P_2q = [6, 30, 60, 90, 120]
arch1_2q = np.array([0.3197, 0.1603, 0.1570, 0.1320, 0.1390]) * 100
arch2_2q = np.array([0.0833, 0.1450, 0.1358, 0.1467, 0.1418]) * 100
arch3_2q = np.array([0.1663, 0.1443, 0.1671, 0.1667, 0.1758]) * 100


# --- Plotting ---

# Set a professional plot style
sns.set_theme(style="whitegrid")
fig,ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('gainsboro')

# Define colors to keep architectures consistent
colors = sns.color_palette("colorblind", 3)

# Plot the 4-Qubit data with SOLID lines (labels removed)
plt.plot(P_4q, arch1_4q, marker='o', linestyle='-', linewidth=4, markersize=8, color=colors[0])
plt.plot(P_4q, arch2_4q, marker='s', linestyle='-', linewidth=4, markersize=8, color=colors[1])
plt.plot(P_4q, arch3_4q, marker='^', linestyle='-', linewidth=4, markersize=10, color=colors[2])

# Plot the 2-Qubit data with DASHED lines (labels removed)
plt.plot(P_2q, arch1_2q, marker='o', linestyle='--', linewidth=4, markersize=8, color=colors[0])
plt.plot(P_2q, arch2_2q, marker='s', linestyle='--', linewidth=4, markersize=8, color=colors[1])
plt.plot(P_2q, arch3_2q, marker='^', linestyle='--', linewidth=4, markersize=10, color=colors[2])


# plt.title('Curvature Ratio Scaling for 2-Qubit and 4-Qubit Ansaetze', fontsize=16)
plt.xlabel('Number of Parameters (P)', fontsize=20)
plt.ylabel(r'$\tilde{L}_{max} / L_{upper}$ (%)', fontsize=20)

# --- Custom Legend ---
# Create a list of custom handles for the legend
legend_elements = [
    # Handles for architecture types (markers only)
    Line2D([0], [0], marker='o', color='w', label='Standard Ring',
           markerfacecolor=colors[0], markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Linear CZ',
           markerfacecolor=colors[1], markersize=8),
    Line2D([0], [0], marker='^', color='w', label='All-to-All CNOT',
           markerfacecolor=colors[2], markersize=10),
    # Handles for qubit counts (lines only)
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='n = 4'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='n = 2')
]

# Apply the custom legend to the plot
plt.legend(handles=legend_elements, title='Ansatz Architecture (Qubits)', fontsize=16, ncol=2, framealpha=1.0, title_fontsize=16)
# --- End Custom Legend Section ---


ax.tick_params(labelsize=20)
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()

# Display the plot
plt.show()