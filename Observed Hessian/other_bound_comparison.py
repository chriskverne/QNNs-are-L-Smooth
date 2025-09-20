import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_qnn(n_layers, n_qubits, n_gates, observable):
    """
    Creates a QNode representing the Variational Quantum Algorithm.

    Args:
        n_layers (int): The number of layers in the ansatz.
        n_qubits (int): The number of qubits.
        n_gates (int): The number of rotation gates per qubit per layer (1, 2, or 3).
        observable (qml.Hamiltonian): The observable to be measured.

    Returns:
        qml.QNode: The quantum circuit.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
    def circuit(params):
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                if n_gates >= 1:
                    qml.RX(params[layer][qubit][0], wires=qubit)
                if n_gates >= 2:
                    qml.RZ(params[layer][qubit][1], wires=qubit)
                if n_gates >= 3:
                    qml.RY(params[layer][qubit][2], wires=qubit)

            # Add entanglement for more complex landscapes
            if n_qubits > 1:
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

        return qml.expval(observable)

    return circuit


def generate_parameter_samples(n_layers, n_qubits, n_gates, n_samples):
    """
    Generates n_samples random parameter tensors.

    Args:
        n_layers (int): The number of layers.
        n_qubits (int): The number of qubits.
        n_gates (int): The number of gates per qubit per layer.
        n_samples (int): The number of random parameter sets to generate.

    Returns:
        pnp.ndarray: An array of random parameter tensors.
    """
    pnp.random.seed(42)
    samples = [pnp.random.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]
    return pnp.array(samples)


def calculate_max_hessian_norm(qnn, samples):
    """
    Calculates the maximum spectral norm of the Hessian across multiple parameter samples.

    Args:
        qnn (qml.QNode): The quantum circuit.
        samples (pnp.ndarray): An array of parameter tensors.

    Returns:
        float: The maximum observed spectral norm of the Hessian.
    """
    max_spectral_norm = 0

    for i, params in enumerate(samples):
        flat_params = params.flatten()

        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(params.shape)
            return qnn(p_reshaped)

        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)

        if spectral_norm > max_spectral_norm:
            max_spectral_norm = spectral_norm

    return max_spectral_norm


# ==================================================================
# Main execution block
# ==================================================================
if __name__ == '__main__':
    # --- Experiment Configuration ---
    n_qubits = 8
    n_layers = 2
    n_gates = 3
    n_samples = 100
    P = n_layers * n_qubits * n_gates

    # --- Define a set of observables to test ---
    observables = {
        "Ising Model": qml.Hamiltonian(
            [-1.0, -0.5, -0.5],
            [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0), qml.PauliX(1)]
        ),
        "Heisenberg Model": qml.Hamiltonian(
            [0.5, 0.5, 0.5],
            [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        ),
        "Mixed-Field Model": qml.Hamiltonian(
            [-1.0, 0.5, 0.5, 0.8, -0.4],
            [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0),qml.PauliZ(1)],
        )
    }

    # --- Data Collection ---
    results = []
    obs_names = []
    for name, obs in observables.items():
        obs_names.append(name)
        print(f"--- Running experiment for: {name} ---")

        # --- Theoretical Bound Calculations ---
        sum_abs_coeffs = sum(abs(c) for c in obs.coeffs)
        hamiltonian_matrix = qml.matrix(obs)
        eigenvalues = pnp.linalg.eigvalsh(hamiltonian_matrix)
        spectral_norm_M = pnp.max(pnp.abs(eigenvalues))

        l_bound_gu = P * sum_abs_coeffs
        l_bound_yours = P * spectral_norm_M

        print(f"P = {P}")
        print(f"Sum |c_k|: {sum_abs_coeffs:.4f}, ||M||_2: {spectral_norm_M:.4f}")
        print(f"Your Bound: {l_bound_yours:.4f}, Gu Bound: {l_bound_gu:.4f}")

        # --- Empirical L_max Calculation ---
        qnn = create_qnn(n_layers, n_qubits, n_gates, obs)
        samples = generate_parameter_samples(n_layers, n_qubits, n_gates, n_samples)
        l_max_empirical = calculate_max_hessian_norm(qnn, samples)
        print(f"Empirical L_max: {l_max_empirical:.4f}\n")

        results.append({
            'l_max': l_max_empirical,
            'gu_bound': l_bound_gu,
            'your_bound': l_bound_yours
        })

    # --- Plotting Results ---
    l_maxs = [r['l_max'] for r in results]
    gu_bounds = [r['gu_bound'] for r in results]
    your_bounds = [r['your_bound'] for r in results]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(obs_names))
    width = 0.25

    rects1 = ax.bar(x - width, gu_bounds, width, label='Gu Bound (P * sum|c_k|)', color='red', alpha=0.8)
    rects2 = ax.bar(x, your_bounds, width, label='Your Bound (P * ||M||_2)', color='green', alpha=0.8)
    rects3 = ax.bar(x + width, l_maxs, width, label='Empirical L_max (Observed)', color='blue', alpha=0.8)

    ax.set_ylabel('L-Smoothness Constant', fontsize=12)
    ax.set_title('Comparison of Hessian Norm Bounds Across Different Observables', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(obs_names, fontsize=11)
    ax.legend(fontsize=12)

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    ax.bar_label(rects3, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.show()

