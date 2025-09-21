# Circuit details: 1 qubit, 20 layers.
# Estimated Maximum Curvature (L_max): 22.6384
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0442

# Circuit details: 1 qubit, 30 layers.
# Estimated Maximum Curvature (L_max): 32.6651
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0306

# Circuit details: 1 qubit, 40 layers.
# Estimated Maximum Curvature (L_max): 44.7352
# Calculated Optimal Learning Rate (η ≈ 1/L_max): 0.0224


import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration for 1-Qubit Experiment ---
N_QUBITS = 1
N_LAYERS = 40  # A deep circuit
N_GATES_PER_ROTATION = 3  # RX, RY, RZ
EPOCHS = 100

# --- Learning Rates to Compare ---
# Based on your data for 40 layers:
ETA_OPTIMAL = 0.0224  # REPLACE with your computed value if different
ETA_HIGH = ETA_OPTIMAL * 5.0
ETA_LOW = 0.001 #ETA_OPTIMAL * 0.2


def create_vqe_circuit(n_layers, n_qubits):
    """
    Creates the VQE circuit for a SINGLE qubit.
    This must be identical to the circuit in the calculation script.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    # A simple 1-qubit Hamiltonian (must match compute_optimal_lr.py)
    coeffs = [0.6, 0.8]
    obs = [qml.PauliX(0), qml.PauliZ(0)]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    # Calculate exact eigenvalue for plotting
    exact_eigenvalue = -pnp.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2)

    @qml.qnode(dev)
    def circuit(params):
        """
        Ansatz for the VQE. Note: No entanglement for n=1.
        """
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RY(params[layer][qubit][1], wires=qubit)
                qml.RZ(params[layer][qubit][2], wires=qubit)
        return qml.expval(hamiltonian)

    return circuit, exact_eigenvalue


def train_vqe(learning_rate, cost_fn):
    """
    Trains a VQE model for a given learning rate and returns the energy history.
    """
    print(f"\n--- Training 1-Qubit VQE with Learning Rate: {learning_rate:.4f} ---")

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
    vqe_circuit, exact_eigenvalue = create_vqe_circuit(N_LAYERS, N_QUBITS)

    history_optimal = train_vqe(ETA_OPTIMAL, vqe_circuit)
    history_high = train_vqe(ETA_HIGH, vqe_circuit)
    history_low = train_vqe(ETA_LOW, vqe_circuit)

    # 3. Plot the results
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    plt.plot(history_optimal, label=f'Optimal η = {ETA_OPTIMAL:.4f}', color='royalblue', linewidth=2.5)
    plt.plot(history_high, label=f'High η = {ETA_HIGH:.4f}', color='indianred', linestyle='--', linewidth=2)
    plt.plot(history_low, label=f'Low η = {ETA_LOW:.4f}', color='mediumseagreen', linestyle=':', linewidth=2)

    plt.axhline(y=exact_eigenvalue, color='black', linestyle='-.', linewidth=1.5,
                label=f'Exact Ground State = {exact_eigenvalue:.4f}')

    plt.title(f'1-Qubit VQE Convergence ({N_LAYERS} Layers)', fontsize=16, fontweight='bold')
    plt.xlabel('Optimization Steps (Epochs)', fontsize=12)
    plt.ylabel('Energy (Expectation Value)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    print("\nDisplaying convergence plot...")
    plt.show()
