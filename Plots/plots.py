import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ================================== DATA =========================================
# 4 qubits, 2 gates, n_layers [1,2,3,4,5,6,7,8,9,10]
fourQ_twoG = [(1, 4, 2, 0.8879927230939715), (2, 4, 2, 0.7149009407554339), (3, 4, 2, 1.118382167660306), (4, 4, 2, 1.4537154108324442), (5, 4, 2, 1.432937049675312), (6, 4, 2, 1.7363925738398756), (7, 4, 2, 2.184481617369816), (8, 4, 2, 2.1005096791101936), (9, 4, 2, 2.519314170603223), (10, 4, 2, 2.8205321708960467)]


data = [(1, 1, 1, 0.9995408577940463), (1, 1, 2, 0.9998505637026228), (1, 1, 3, 1.965525726370158), (1, 2, 1, 0.997249393360669), (1, 2, 2, 0.986007080865856), (1, 2, 3, 1.9182122556740817), (1, 4, 1, 0.7951830635810667), (1, 4, 2, 0.8879927230939715), (1, 4, 3, 1.1836489427456423), (1, 6, 1, 0.6448097102754932), (1, 6, 2, 0.5699959336731839), (1, 6, 3, 0.8538882124916154), (1, 8, 1, 0.39002983947396247), (1, 8, 2, 0.4380778457243751), (1, 8, 3, 0.7682947642956021), (2, 1, 1, 1.9999996875492156), (2, 1, 2, 1.975936518903822), (2, 1, 3, 2.961160669069182), (2, 2, 1, 1.6269865381465467), (2, 2, 2, 1.8466691277421716), (2, 2, 3, 2.4071741737372507), (2, 4, 1, 0.7215423516447204), (2, 4, 2, 0.7149009407554339), (2, 4, 3, 1.3624049124275992), (2, 6, 1, 0.42105495298888007), (2, 6, 2, 0.4681995043285483), (2, 6, 3, 1.0505349887438216), (2, 8, 1, 0.25148438225657765), (2, 8, 2, 0.26729427076705997), (2, 8, 3, 0.5068252216608563), (3, 1, 1, 2.9994617642137635), (3, 1, 2, 2.7168769901376644), (3, 1, 3, 4.458701586751792), (3, 2, 1, 1.9055053593592666), (3, 2, 2, 2.5314174721714404), (3, 2, 3, 3.5236568840002587), (3, 4, 1, 0.7150244879027736), (3, 4, 2, 1.118382167660306), (3, 4, 3, 1.7315430795542082), (3, 6, 1, 0.35030566294702203), (3, 6, 2, 0.4786544806342642), (3, 6, 3, 0.9160481434349866), (3, 8, 1, 0.1936444391304041), (3, 8, 2, 0.32466597511889955), (3, 8, 3, 0.6432356774512358)]


df = pd.DataFrame(data, columns=['n_layers', 'n_qubits', 'n_gates', 'L_eff'])
df['P'] = df['n_layers'] * df['n_qubits'] * df['n_gates']

# =================================== PLOTTING =======================================
plt.figure(figsize=(12, 8))
plt.plot([0, df['P'].max()], [0, df['P'].max()], color='crimson', linestyle='--', linewidth=3.5, label='Bound ($L_{bound} \leq P$)')

qubit_colors = {
    1: '#1f77b4',  # Blue
    2: '#ff7f0e',  # Orange
    4: '#2ca02c',  # Green
    6: '#d62728',  # Red
    8: '#9467bd'   # Purple
}

# 3. Create scatter plots for each qubit count with distinct colors
for n_qubits in sorted(df['n_qubits'].unique()):
    subset = df[df['n_qubits'] == n_qubits]
    plt.scatter(subset['P'], subset['L_eff'],
               c=qubit_colors[n_qubits],
               s=200,
               alpha=1,
               label=f'{n_qubits} Qubit(s)',
               zorder=5)

    m, b = np.polyfit(subset['P'], subset['L_eff'], 1)

    # Generate x-values for the regression line to span the range of P for the current subset.
    p_vals = np.array([subset['P'].min(), subset['P'].max()])
    p_vals = np.array([0, 100])

    # Plot the regression line using the calculated slope and intercept.
    plt.plot(p_vals, m * p_vals + b,
             color=qubit_colors[n_qubits],
             linestyle='-',
             linewidth=2.5,
             label=f'{n_qubits} Qubit(s) Trend')

# 4. Final plot adjustments for clarity and publication quality
plt.xlabel('Total Number of Parameters (P)', fontsize=16)
plt.ylabel('Largest Measured Hessian Norm ($L_{eff}$)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(framealpha=1.0, facecolor='white', edgecolor='black', fontsize=16)
plt.xlim(left=0)
plt.ylim(0, 5)
plt.xlim(0, 75)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.gcf().set_facecolor("white")
plt.gca().set_facecolor("#e2ddddaf")
plt.show()