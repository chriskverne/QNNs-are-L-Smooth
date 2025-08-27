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
        # coeffs = [1.0, 1.2]
        # obs = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)]
        # observable = qml.Hamiltonian(coeffs, obs)

        return qml.expval(observable)

    return circuit

def generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=2):
    """
    generates n_samples random parameter tensors
    """
    pnp.random.seed(42)
    samples = [pnp.random.uniform(0, 2*pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]
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
        #eigenvalues = pnp.linalg.eigvalsh(hessian_matrix)
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
    results_data = []

    """
    Experiment 1: Fix N_qubits, N_Gates, Increase N_Layers
    Either 2,3 or 4 qubits sounds good
    N_Gates = 2 or 3 sounds good
    """
    n_samples = 100
    n_gates = 3
    n_qubits = 10
    n_layer_combos = [1,2,3,4] # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    entanglement = False

    for n_layers in n_layer_combos:
        # --- QNN and Data Generation ---
        qnn = create_qnn(n_layers, n_qubits, n_gates, entangled=entanglement)
        samples = generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=n_gates)

        # --- Theoretical Bound Calculation ---
        P = n_layers * n_qubits * n_gates
        norm_M = 1.0
        L_bound = P * norm_M

        # --- Experiment ---
        hessian_norms = calculate_hessian_norms(qnn, samples)

        # --- Visualization ---
        #plot_results(hessian_norms, L_bound, P)

        # --- Verification ---
        all_within_bound = all(norm <= L_bound for norm in hessian_norms)
        print("--- Experiment Setup ---")
        print(f"Number of Layers: {n_layers}, Number of Qubits: {n_qubits}, Number of Gates: {n_gates}, Total Parameters (P): {P}")
        print(f"Theoretical L-Smoothness Bound (L <= P): {L_bound:.4f}")
        print(f"Largest Hessian Norm: {pnp.max(hessian_norms)}")
        results_data.append((n_layers, n_qubits, n_gates, pnp.max(hessian_norms)))

    print(results_data)
