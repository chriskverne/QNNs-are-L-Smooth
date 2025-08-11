import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns

def create_qnn(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RZ(params[layer][qubit][1], wires=qubit)

            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CNOT(wires=[qubit, next_qubit])

        return  qml.expval(qml.PauliZ(0))

    return circuit

def generate_parameter_samples(n_layers, n_qubits, n_samples):
    """
    generates n_samples random parameter tensors
    """
    pnp.random.seed(42)
    samples = [pnp.random.uniform(0, 2*pnp.pi, size=(n_layers, n_qubits, 2)) for _ in range(n_samples)]
    return pnp.array(samples)

def calculate_hessian_norms(qnn, samples):
    hessian_norms = []
    hessian_fn = qml.jacobian(qml.grad(qnn))
    for i, params in enumerate(samples):
        # Output of QNN only accepts flat parameter vector
        flat_params = params.flatten()
        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(params.shape)
            return qnn(p_reshaped)

        # Calculate hessian matrix
        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)

        # Calculate the spectral norm (largest singular value) of the Hessian = largest absolute eigenvalue.
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        hessian_norms.append(spectral_norm)

    return hessian_norms


def plot_results(hessian_norms, bound, P):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # Plot the individual Hessian norms
    plt.scatter(range(len(hessian_norms)), hessian_norms, label='Experimental Hessian Norm', color='royalblue',
                zorder=5)
    # Plot the theoretical bound
    plt.axhline(y=bound, color='crimson', linestyle='--', linewidth=2, label=f'Theoretical Bound (L = P = {P})')

    plt.title('Experimental Verification of L-Smoothness Bound', fontsize=16)
    plt.xlabel('Parameter Sample Index', fontsize=12)
    plt.ylabel('Spectral Norm of Hessian', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# ==================================================================
# Main execution block
# ==================================================================
if __name__ == '__main__':
    # --- Configuration ---
    n_qubits = 4
    n_layers = 2
    n_samples = 20

    # --- Theoretical Bound Calculation ---
    P = n_layers * n_qubits * 2
    norm_M = 1.0
    L_bound = P * norm_M

    print("--- Experiment Setup ---")
    print(f"Number of Qubits: {n_qubits}")
    print(f"Number of Layers: {n_layers}")
    print(f"Total Parameters (P): {P}")
    print(f"Observable: PauliZ(0), Norm ||M||_2 = {norm_M}")
    print(f"Theoretical L-Smoothness Bound (L <= P): {L_bound:.4f}\n")

    # --- QNN and Data Generation ---
    qnn = create_qnn(n_layers, n_qubits)
    samples = generate_parameter_samples(n_layers, n_qubits, n_samples)

    # --- Experiment ---
    hessian_norms = calculate_hessian_norms(qnn, samples)

    # --- Visualization ---
    plot_results(hessian_norms, L_bound, P)

    # --- Verification ---
    all_within_bound = all(norm <= L_bound for norm in hessian_norms)
    print("\n--- Verification Result ---")
    print(f"All calculated Hessian norms are within the theoretical bound: {all_within_bound}")

