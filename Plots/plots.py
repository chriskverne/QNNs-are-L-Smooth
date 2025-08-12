import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data provided by the user, representing the experimental results.
# Each tuple is in the format: (Layers, Qubits, Gates, Largest_Hessian_Norm)
data = [
    (1, 1, 1, 0.9995408577940463), (1, 1, 2, 0.9998505637026228), (1, 2, 1, 0.997249393360669),
    (1, 2, 2, 0.986007080865856), (1, 2, 3, 1.9182122556740817), (1, 4, 1, 0.7951830635810667),
    (1, 4, 2, 0.8879927230939715), (1, 4, 3, 1.1836489427456423), (1, 8, 1, 0.39002983947396247),
    (1, 8, 2, 0.4380778457243751), (1, 8, 3, 0.7682947642956021), (2, 1, 1, 1.9999996875492156),
    (2, 1, 2, 1.975936518903822), (2, 1, 3, 2.961160669069182), (2, 2, 1, 1.6269865381465467),
    (2, 2, 2, 1.8466691277421716), (2, 2, 3, 2.4071741737372507), (2, 4, 1, 0.7215423516447204),
    (2, 4, 2, 0.7149009407554339), (2, 4, 3, 1.3624049124275992), (2, 8, 1, 0.25148438225657765),
    (2, 8, 2, 0.26729427076705997), (2, 8, 3, 0.5068252216608563), (3, 1, 1, 2.9994617642137635),
    (3, 1, 2, 2.7168769901376644), (3, 1, 3, 4.458701586751792), (3, 2, 1, 1.9055053593592666),
    (3, 2, 2, 2.5314174721714404), (3, 2, 3, 3.5236568840002587), (3, 4, 1, 0.7150244879027736),
    (3, 4, 2, 1.118382167660306), (3, 4, 3, 1.7315430795542082), (3, 8, 1, 0.1936444391304041),
    (3, 8, 2, 0.32466597511889955), (3, 8, 3, 0.6432356774512358)
]

# Create a pandas DataFrame for easier data manipulation
df = pd.DataFrame(data, columns=['n_layers', 'n_qubits', 'n_gates', 'L_eff'])
df['P'] = df['n_layers'] * df['n_qubits'] * df['n_gates']

# Set the theme for the plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# 1. Plot the theoretical bound (y = x line)
max_p = df['P'].max()
plt.plot([0, max_p], [0, max_p], color='crimson', linestyle='--', linewidth=2.5, label='Theoretical Bound ($L_{bound} = P$)')

# 2. Create the scatter plot of the empirical data
scatter = plt.scatter(df['P'], df['L_eff'], c=df['n_qubits'], cmap='viridis', s=100, alpha=0.8, zorder=5)

# 3. Add a colorbar to explain what the colors mean
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Qubits (n)', fontsize=14, rotation=270, labelpad=20)
cbar.set_ticks(df['n_qubits'].unique()) # Ensures ticks are on 1, 2, 4, 8

# 4. Final plot adjustments for clarity and publication quality
plt.xlabel('Total Number of Parameters (P)', fontsize=16)
plt.ylabel('Largest Measured Hessian Norm ($L_{eff}$)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.ylim(0,5)
plt.show()