import itertools
import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def create_qnn_two(n_layers, n_qubits, n_gates, observable_coeffs, observable_ops, entangled=True):
    """
    Creates a QNN with a "Strongly Entangling Layers" architecture.
    - Rotations: RX, RY, RZ gates.
    - Entanglement: CZ gates in a linear chain topology.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for layer in range(n_layers):
            # 1. Parameterized rotation layer
            for qubit in range(n_qubits):
                if n_gates == 1:
                    qml.RX(params[layer][qubit][0], wires=qubit)
                elif n_gates == 2:
                    qml.RY(params[layer][qubit][0], wires=qubit)
                    qml.RZ(params[layer][qubit][1], wires=qubit)
                elif n_gates == 3:
                    qml.RX(params[layer][qubit][0], wires=qubit)
                    qml.RY(params[layer][qubit][1], wires=qubit)
                    qml.RZ(params[layer][qubit][2], wires=qubit)

            # 2. Entanglement layer
            if entangled and n_qubits > 1:
                # Use CZ gates in a linear chain (i -> i+1)
                for qubit in range(n_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])

        observable = qml.Hamiltonian(observable_coeffs, observable_ops)
        return qml.expval(observable)

    return circuit

def create_qnn_three(n_layers, n_qubits, n_gates, observable_coeffs, observable_ops, entangled=True):
    """
    Creates a QNN with an "All-to-All" entanglement structure.
    - Rotations: RZ, RY, RZ gates (a different universal set).
    - Entanglement: CNOT gates applied to every unique pair of qubits.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for layer in range(n_layers):
            # 1. Parameterized rotation layer with a different gate sequence
            for qubit in range(n_qubits):
                if n_gates == 1:
                    qml.RY(params[layer][qubit][0], wires=qubit)
                elif n_gates == 2:
                    qml.RY(params[layer][qubit][0], wires=qubit)
                    qml.RZ(params[layer][qubit][1], wires=qubit)
                elif n_gates == 3:
                    # Using a different universal rotation sequence (Euler angles)
                    qml.RZ(params[layer][qubit][0], wires=qubit)
                    qml.RY(params[layer][qubit][1], wires=qubit)
                    qml.RZ(params[layer][qubit][2], wires=qubit)

            # 2. Dense, all-to-all entanglement layer
            if entangled and n_qubits > 1:
                # Apply a CNOT to every possible pair of qubits
                for q1, q2 in itertools.combinations(range(n_qubits), 2):
                    qml.CNOT(wires=[q1, q2])

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

# ==================================================================
# Main execution block
# ==================================================================
if __name__ == '__main__':
    all_results = []

    # --- Experiment Configuration ---
    n_samples = 100 # Use 1000 for final run
    n_qubits = 4  # 4 qubits is a great choice
    n_gates = 3  # All architectures will use 3 gates per qubit per layer

    architectures = {
        "Standard Ring (QNN 1)": create_qnn,
        "Linear CZ (QNN 2)": create_qnn_two,
        "All-to-All CNOT (QNN 3)": create_qnn_three
    }

    # P = n_layers * 4 qubits * 3 gates = 12 * n_layers.
    # This combo reaches P=120 at 10 layers.
    layer_combos = [1, 2, 4, 6, 8, 10]

    for arch_name, qnn_func in architectures.items():
        print(f"\n{'=' * 20} Running Architecture: {arch_name} {'=' * 20}")

        observable_coeffs = [1 / n_qubits] * n_qubits
        observable_ops = [qml.PauliZ(i) for i in range(n_qubits)]
        norm_M = 1.0

        for n_layers in layer_combos:
            P = n_layers * n_qubits * n_gates

            print(f"  Testing {n_layers} layers (P={P})...")

            qnn = qnn_func(n_layers, n_qubits, n_gates,
                           observable_coeffs, observable_ops, entangled=True)

            samples = generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=n_gates)

            L_bound = P * norm_M
            hessian_norms = calculate_hessian_norms(qnn, samples)
            max_hessian_norm = pnp.max(hessian_norms)

            ratio = (max_hessian_norm / L_bound * 100) if L_bound > 0 else 0
            all_results.append({
                "Architecture": arch_name,
                "P": P,
                "Ratio": ratio
            })
            print(f"    -> L_max = {max_hessian_norm:.4f}, L_upper = {L_bound:.4f}, Ratio = {ratio / 100:.4f}")

    # --- Process and Plot Results ---
    results_df = pd.DataFrame(all_results)

    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=results_df, x="P", y="Ratio", hue="Architecture", marker="o", style="Architecture")
    # plt.xlabel("Number of Parameters (P)")
    # plt.ylabel(r"$\tilde{L}_{max} / L_{upper}$ (%)")
    # plt.title(f"Curvature Ratio Scaling for Different {n_qubits}-Qubit Ansaetze")
    # plt.grid(True, which='both', linestyle='--')
    # plt.legend(title="Ansatz Architecture")
    # plt.tight_layout()
    # plt.savefig("ansatz_comparison_final.png")
    # plt.show()