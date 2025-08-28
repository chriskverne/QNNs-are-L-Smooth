import itertools
import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns

def create_qnn(n_layers, n_qubits, n_gates, observable_coeffs, observable_ops, entangled=True):
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

        # observable = qml.Hamiltonian([1 / n_qubits] * n_qubits, [qml.PauliZ(i) for i in range(n_qubits)])
        observable = qml.Hamiltonian(observable_coeffs, observable_ops)
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

    """
    Experiment 1: Fix N_qubits, N_Gates, Increase N_Layers
    Either 2,3 or 4 qubits sounds good
    N_Gates = 2 or 3 sounds good
    """
    n_samples = 100
    n_layers = 2
    n_qubits = 8
    n_gates = 3
    entanglement = False
    #observable_ops = [qml.PauliZ(0), qml.PauliX(1)]
    observable_ops = [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0),
        qml.PauliX(1)
    ]

    experiment_results = []
    weights_to_test = pnp.linspace(0.1, 5.0, 10) # Use 15 points for a smoother plot
    samples = generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=n_gates)

    for w in weights_to_test:
        # --- Define the observable for this iteration ---
        #coeffs = pnp.array([w, 2.0])  # Using a fixed second coeff to make norm non-trivial (we can try adjusting this later)
        coeffs = pnp.array([1.0, w, w])

        # Calculate the theoretical spectral norm of M
        temp_hamiltonian = qml.Hamiltonian(coeffs, observable_ops)
        norm_M = pnp.max(pnp.abs(temp_hamiltonian.eigvals()))

        # --- QNN and Bound Calculation for this M ---
        qnn = create_qnn(n_layers, n_qubits, n_gates, coeffs, observable_ops, entangled=entanglement)
        L_bound = n_layers * n_qubits * n_gates * norm_M

        # --- Run Experiment ---
        hessian_norms = calculate_hessian_norms(qnn, samples)
        max_measured_norm = pnp.max(hessian_norms)

        # --- Store and Print Results ---
        experiment_results.append((norm_M.item(), max_measured_norm.item(), L_bound.item()))
        print(f"Weight w={w:.2f} -> ||M||_2={norm_M:.4f}, L_bound={L_bound:.4f}, L_max={max_measured_norm:.4f}")

    print(experiment_results)

# M = z0*z1 + w (x0 + x1) then adjust w
# n_qubits = 4, n_layers = 2, n_gates = 3