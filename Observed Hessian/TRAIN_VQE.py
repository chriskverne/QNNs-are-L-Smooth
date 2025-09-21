# Circuit details: 1 qubit, 20 layers.
# Estimated Maximum Curvature (L_max): 22.6384
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0442

# Circuit details: 1 qubit, 30 layers.
# Estimated Maximum Curvature (L_max): 32.6651
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0306

# Circuit details: 1 qubit, 40 layers.
# Estimated Maximum Curvature (L_max): 44.7352
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0224

# N_QUBITS = 2
# N_LAYERS = 5
# N_GATES_PER_ROTATION = 3
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.1062

# N_QUBITS = 2
# N_LAYERS = 10
# N_GATES_PER_ROTATION = 3
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0641

# Circuit details: 2 qubits, 15 layers.
# Estimated Maximum Curvature (L_max): 19.9360
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0502

# N_QUBITS = 2
# N_LAYERS = 20
# N_GATES_PER_ROTATION = 3
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0379

# N_QUBITS = 4
# N_LAYERS = 2
# N_GATES_PER_ROTATION = 3
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.2886

# N_QUBITS = 4
# N_LAYERS = 5
# N_GATES_PER_ROTATION = 3
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.1248

# Circuit details: 4 qubits, 6 layers.
# Estimated Maximum Curvature (L_max): 8.8849
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.1126

# Circuit details: 4 qubits, 8 layers.
# Estimated Maximum Curvature (L_max): 12.8617
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0778

# Circuit details: 4 qubits, 9 layers.
# Estimated Maximum Curvature (L_max): 13.1633
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0760

# Circuit details: 4 qubits, 10 layers.
# Estimated Maximum Curvature (L_max): 16.3505
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0612

# Circuit details: 8 qubits, 1 layers.
# Estimated Maximum Curvature (L_max): 2.0022
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.4995

# Circuit details: 8 qubits, 3 layers.
# Estimated Maximum Curvature (L_max): 5.9923
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.1669

# Circuit details: 8 qubits, 5 layers.
# Estimated Maximum Curvature (L_max): 6.8638
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.1457

# Circuit details: 10 qubits, 1 layers.
# Estimated Maximum Curvature (L_max): 2.0169
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.4958

# Circuit details: 10 qubits, 2 layers.
# Estimated Maximum Curvature (L_max): 3.9519
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.2530

# Circuit details: 10 qubits, 4 layers.
# Estimated Maximum Curvature (L_max): 5.7854
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.1729

import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
N_QUBITS = 4
N_LAYERS = 10  # Increased from 4 to make the landscape more complex
N_GATES_PER_ROTATION = 3  # RX, RY, RZ
EPOCHS = 150  # Increased to allow for longer convergence time

# --- Learning Rates to Compare ---
# This value MUST be taken from the output of the updated 'compute_optimal_lr.py'.
ETA_OPTIMAL = 0.0612  # Placeholder: REPLACE with your newly computed value
ETA_HIGH = ETA_OPTIMAL * 5.0  # Increased multiplier to show more instability
ETA_LOW = ETA_OPTIMAL * 0.2  # Decreased multiplier to show slower convergence
ETA_STANDARD = 0.001

def create_vqe_circuit(n_layers, n_qubits):
    """
    Creates the VQE circuit (QNode) with a defined ansatz and Hamiltonian.
    This must be identical to the circuit in the calculation script.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    # Define the Hamiltonian for a general 1D transverse-field Ising model
    # H = - sum(Z_i @ Z_{i+1}) + 0.5 * sum(X_i)
    obs = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n_qubits - 1)]
    obs.extend([qml.PauliX(i) for i in range(n_qubits)])

    coeffs = [-1.0] * (n_qubits - 1)
    coeffs.extend([0.5] * n_qubits)

    hamiltonian = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev)
    def circuit(params):
        """
        Ansatz for the VQE, using layers of rotations and entanglement.
        """
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RY(params[layer][qubit][1], wires=qubit)
                qml.RZ(params[layer][qubit][2], wires=qubit)
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        return qml.expval(hamiltonian)

    return circuit, hamiltonian


def train_vqe(learning_rate, cost_fn):
    """
    Trains a VQE model for a given learning rate and returns the energy history.
    """
    print(f"\n--- Training VQE with Learning Rate: {learning_rate:.4f} ---")

    pnp.random.seed(42)  # Use the same seed for fair comparison
    param_shape = (N_LAYERS, N_QUBITS, N_GATES_PER_ROTATION)
    initial_params = pnp.random.uniform(0, 2 * pnp.pi, size=param_shape, requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    params = initial_params
    energy_history = []

    for epoch in range(EPOCHS):
        params, cost = optimizer.step_and_cost(cost_fn, params)
        energy_history.append(cost)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}: Cost = {cost:.8f}")

    return energy_history


if __name__ == '__main__':
    # 1. Create the VQE circuit and cost function
    vqe_circuit, hamiltonian = create_vqe_circuit(N_LAYERS, N_QUBITS)

    # Get the exact ground state energy for reference on the plot
    exact_eigenvalue = qml.eigvals(hamiltonian)[0]

    # 2. Train VQE with three different learning rates
    history_optimal = train_vqe(ETA_OPTIMAL, vqe_circuit)
    history_high = train_vqe(ETA_HIGH, vqe_circuit)
    history_low = train_vqe(ETA_LOW, vqe_circuit)
    history_standard = train_vqe(ETA_STANDARD, vqe_circuit)

    # 3. Plot the results
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    plt.plot(history_optimal, label=f'Optimal η ≈ {ETA_OPTIMAL:.4f}', color='royalblue', linewidth=2.5)
    plt.plot(history_high, label=f'High η = {ETA_HIGH:.4f}', color='indianred', linestyle='--', linewidth=2)
    plt.plot(history_low, label=f'Low η = {ETA_LOW:.4f}', color='Purple', linewidth=2)
    plt.plot(history_standard, label=f'Standard η = {ETA_STANDARD:.4f}', color='mediumseagreen', linestyle=':', linewidth=2)

    print(history_optimal)
    print(history_high)
    print(history_low)
    print(history_standard)

    # Plot the exact ground state energy as a horizontal line
    plt.axhline(y=exact_eigenvalue, color='black', linestyle='-.', linewidth=1.5,
                label=f'Exact Ground State = {exact_eigenvalue:.4f}')

    plt.title('VQE Convergence with a Deeper Circuit (15 Layers)', fontsize=16, fontweight='bold')
    plt.xlabel('Optimization Steps (Epochs)', fontsize=12)
    plt.ylabel('Energy (Expectation Value)', fontsize=12)
    plt.xlim(0,40)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    print("\nDisplaying convergence plot...")
    plt.show()


