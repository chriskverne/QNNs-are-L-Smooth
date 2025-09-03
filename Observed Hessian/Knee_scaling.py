import itertools
import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import pandas as pd


# ==================================================================
# Original functions from your code
# ==================================================================

def create_qnn_expval(n_layers, n_qubits, n_gates, observable_coeffs, observable_ops, entangled=True):
    """
    Creates a QNN that returns an expectation value.
    This is used for the Hessian calculation.
    """
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
                # Use a linear chain of CNOTs for entanglement
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

        observable = qml.Hamiltonian(observable_coeffs, observable_ops)
        return qml.expval(observable)

    return circuit


def generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=2):
    #pnp.random.seed(42)
    samples = [pnp.random.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]
    return pnp.array(samples)


def calculate_hessian_norms(qnn, samples):
    hessian_norms = []
    for i, params in enumerate(samples):
        flat_params = params.flatten()

        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(params.shape)
            return qnn(p_reshaped)

        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        hessian_norms.append(spectral_norm)
    return hessian_norms


# ==================================================================
# NEW functions based on Sim et al. to calculate expressibility
# ==================================================================

def create_qnn_state(n_layers, n_qubits, n_gates, entangled=True):
    """
    Creates a QNN that returns the final state vector.
    This is needed to calculate fidelities for the expressibility metric.
    """
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
                # Use a linear chain of CNOTs for entanglement
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
        return qml.state()

    return circuit


def calculate_expressibility_kld(qnn_state, n_layers, n_qubits, n_gates, n_fidelity_samples=1000):
    """
    Calculates the expressibility metric (KL Divergence) as defined in Sim et al.
    """
    # Generate parameter sets for pairs of states
    param_samples_1 = generate_parameter_samples(n_layers, n_qubits, n_fidelity_samples, n_gates)
    param_samples_2 = generate_parameter_samples(n_layers, n_qubits, n_fidelity_samples, n_gates)

    # Calculate fidelities between pairs of generated states
    fidelities = []
    for p1, p2 in zip(param_samples_1, param_samples_2):
        state1 = qnn_state(p1)
        state2 = qnn_state(p2)

        # --- FIX IS HERE ---
        fidelity = pnp.abs(pnp.vdot(state1, state2)) ** 2
        fidelities.append(fidelity)

    # 1. Estimate the PQC's fidelity distribution with a histogram
    n_bins = 75
    bins = pnp.linspace(0, 1, n_bins + 1)
    p_pqc, _ = pnp.histogram(fidelities, bins=bins, density=True)
    p_pqc = p_pqc + 1e-10

    # 2. Calculate the analytical Haar distribution for the same bins
    N = 2 ** n_qubits
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p_haar = (N - 1) * (1 - bin_centers) ** (N - 2)
    p_haar /= pnp.sum(p_haar)
    p_haar = p_haar + 1e-10

    # 3. Compute the KL Divergence
    kld = entropy(p_pqc, p_haar)
    return kld


# ==================================================================
# Main execution block
# ==================================================================
if __name__ == '__main__':
    pnp.random.seed(42)
    results_data = []

    # --- Experiment Configuration ---
    n_hessian_samples = 100
    n_fidelity_samples = 2000
    n_gates = 2
    n_qubits = 4
    n_layer_combos = range(1, 11)
    entanglement = True

    observable_coeffs = [1.0]
    observable_ops = [qml.PauliZ(0)]

    print("Starting experiment: Correlating Hessian Norm with Expressibility")
    print(f"Qubits: {n_qubits}, Gates per qubit/layer: {n_gates}, Entangled: {entanglement}")
    print("-" * 60)

    for n_layers in n_layer_combos:
        P = n_layers * n_qubits * n_gates

        qnn_expval = create_qnn_expval(n_layers, n_qubits, n_gates, observable_coeffs, observable_ops,
                                       entangled=entanglement)
        hessian_samples = generate_parameter_samples(n_layers, n_qubits, n_hessian_samples, n_gates=n_gates)
        hessian_norms = calculate_hessian_norms(qnn_expval, hessian_samples)
        max_hessian_norm = pnp.max(hessian_norms)

        norm_M = 1.0
        L_bound = P * norm_M
        l_max_ratio = max_hessian_norm / L_bound if L_bound > 0 else 0

        qnn_state = create_qnn_state(n_layers, n_qubits, n_gates, entangled=entanglement)
        kld = calculate_expressibility_kld(qnn_state, n_layers, n_qubits, n_gates,
                                           n_fidelity_samples=n_fidelity_samples)

        results_data.append({
            'layers': n_layers,
            'P': P,
            'L_max_ratio': l_max_ratio,
            'D_KL': kld
        })
        print(
            f"Layers: {n_layers:2d} | P: {P:3d} | L_max/L_upper: {l_max_ratio:.4f} | Expressibility (D_KL): {kld:.4f}")

    print(results_data)
    # --- Plotting the Results ---
    df = pd.DataFrame(results_data)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Parameters (P)')
    ax1.set_ylabel('Hessian Norm Ratio ($L_{max}/L_{upper}$)', color=color)
    ax1.plot(df['P'], df['L_max_ratio'], marker='o', linestyle='-', color=color, label='Hessian Norm Ratio')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Expressibility ($D_{KL}$)', color=color)
    ax2.plot(df['P'], df['D_KL'], marker='x', linestyle='--', color=color, label='Expressibility (Sim et al.)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f'Verifying Expressibility Saturation for {n_qubits}-Qubit QNN')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.show()