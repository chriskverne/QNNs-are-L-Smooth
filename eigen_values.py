import itertools
import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns


def create_qnn(n_layers, n_qubits, n_gates, entangled=True):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                if n_gates == 1:
                    qml.RX(params[layer][qubit][0], wires=qubit)
                elif n_gates == 2:
                    qml.RX(params[layer][qubit][0], wires=qubit)
                    qml.RZ(params[layer][qubit][1], wires=qubit)
                elif n_gates == 3:
                    qml.RX(params[layer][qubit][0], wires=qubit)
                    qml.RZ(params[layer][qubit][1], wires=qubit)
                    qml.RY(params[layer][qubit][2], wires=qubit)

            if entangled:
                for qubit in range(n_qubits):
                    if n_qubits <= 1:
                        continue
                    next_qubit = (qubit + 1) % n_qubits
                    qml.CNOT(wires=[qubit, next_qubit])

        observable = qml.Hamiltonian(
            [1 / n_qubits] * n_qubits,
            [qml.PauliZ(i) for i in range(n_qubits)]
        )
        return qml.expval(observable)

    return circuit


def generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=3):
    """
    Generates n_samples random parameter tensors.
    """
    pnp.random.seed(42)
    samples = [pnp.random.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]
    return pnp.array(samples)


def calculate_hessian_eigenvalues(qnn, samples):
    """
    Calculates the Hessian for each parameter sample and collects all eigenvalues.

    Returns:
        A flat numpy array containing all eigenvalues from all Hessian matrices.
    """
    all_eigenvalues = []
    for i, params in enumerate(samples):
        # Flatten parameters for PennyLane's gradient functions
        flat_params = params.flatten()

        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(params.shape)
            return qnn(p_reshaped)

        # Calculate the full Hessian matrix
        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)

        # Calculate the eigenvalues of the symmetric Hessian matrix
        # eigvalsh is efficient for real, symmetric matrices.
        eigenvalues = pnp.linalg.eigvalsh(hessian_matrix)
        all_eigenvalues.extend(eigenvalues)

        print(f"Processed sample {i + 1}/{len(samples)}...")

    return pnp.array(all_eigenvalues)


def plot_eigenvalue_distribution(eigenvalues, bound, P, n_qubits, n_layers):
    """
    Plots a histogram of the Hessian eigenvalues and the theoretical bounds.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    sns.histplot(eigenvalues, kde=True, bins=50, stat="density")

    # Plot the theoretical bounds for the eigenvalues
    plt.axvline(x=bound, color='crimson', linestyle='--', linewidth=2, label=f'Upper Bound (+L = {bound})')
    plt.axvline(x=-bound, color='crimson', linestyle='--', linewidth=2, label=f'Lower Bound (-L = {-bound})')

    plt.title(f'Hessian Eigenvalue Distribution (n_qubits={n_qubits}, n_layers={n_layers}, P={P})', fontsize=16)
    plt.xlabel('Eigenvalue ($\lambda$)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ==================================================================
# Main execution block
# ==================================================================
if __name__ == '__main__':
    # --- Experiment Configuration ---
    n_samples = 20
    n_gates = 3
    n_qubits = 4
    n_layers = 2
    entanglement = False

    # --- QNN and Data Generation ---
    qnn = create_qnn(n_layers, n_qubits, n_gates, entangled=entanglement)
    samples = generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=n_gates)

    # --- Theoretical Bound Calculation ---
    P = n_layers * n_qubits * n_gates
    norm_M = 1.0  # As per your simplified bound for Pauli Z observables
    L_bound = P * norm_M

    # --- Experiment: Calculate all eigenvalues ---
    all_eigenvalues = calculate_hessian_eigenvalues(qnn, samples)

    # --- Analysis & Verification ---
    max_abs_eigenvalue = pnp.max(pnp.abs(all_eigenvalues))
    all_within_bound = max_abs_eigenvalue <= L_bound

    print("\n--- Experiment Summary ---")
    print(f"Number of Layers: {n_layers}, Number of Qubits: {n_qubits}, Parameters (P): {P}")
    print(f"Theoretical L-Smoothness Bound (L <= P): {L_bound:.4f}")
    print(f"Largest observed absolute eigenvalue (Spectral Norm): {max_abs_eigenvalue:.4f}")
    print(f"Smallest observed eigenvalue: {pnp.min(all_eigenvalues):.4f}")
    print(f"Mean of eigenvalues: {pnp.mean(all_eigenvalues):.4f}")
    print(f"All eigenvalues within the bound [-L, L]? {all_within_bound}")

    # --- Visualization ---
    plot_eigenvalue_distribution(all_eigenvalues, L_bound, P, n_qubits, n_layers)