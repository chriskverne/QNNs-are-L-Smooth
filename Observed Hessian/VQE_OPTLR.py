import pennylane as qml
from pennylane import numpy as pnp
import itertools

# --- Configuration ---
N_QUBITS = 4
N_LAYERS = 6 # Increased from 4 to 15
N_GATES_PER_ROTATION = 3 # RX, RY, RZ
N_SAMPLES = 200 # Reduced to speed up calculation for the larger circuit

def create_vqe_circuit(n_layers, n_qubits):
    """
    Creates the VQE circuit (QNode) with a defined ansatz and Hamiltonian.
    This defines the problem we are trying to solve.
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
            # Layer of rotational gates
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RY(params[layer][qubit][1], wires=qubit)
                qml.RZ(params[layer][qubit][2], wires=qubit)

            # Entanglement layer
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return qml.expval(hamiltonian)

    return circuit, hamiltonian


def generate_parameter_samples(n_layers, n_qubits, n_samples):
    """
    Generates n_samples random parameter tensors for exploring the landscape.
    """
    # pnp.random.seed(42)
    param_shape = (n_layers, n_qubits, N_GATES_PER_ROTATION)
    samples = [pnp.random.uniform(0, 2 * pnp.pi, size=param_shape) for _ in range(n_samples)]
    return pnp.array(samples)


def calculate_max_hessian_norm(qnode, samples):
    """
    Calculates the spectral norm of the Hessian for each parameter sample
    and returns the maximum observed value (our estimate for L_max).
    """
    hessian_norms = []
    total_params = samples[0].size
    param_shape = samples[0].shape

    print(f"Starting Hessian norm calculation for {len(samples)} samples...")
    for i, params in enumerate(samples):
        # The qml.jacobian function requires a cost function that accepts a flat vector
        flat_params = params.flatten()
        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(param_shape)
            return qnode(p_reshaped)

        # Calculate the Hessian matrix
        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)

        # Calculate the spectral norm (largest singular value)
        # For a symmetric matrix like the Hessian, this is the largest absolute eigenvalue.
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        hessian_norms.append(spectral_norm)

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i+1}/{len(samples)} samples.")

    return pnp.max(hessian_norms)


if __name__ == '__main__':
    print("--- VQE Landscape Analysis for Optimal Learning Rate ---")
    print(f"Circuit details: {N_QUBITS} qubits, {N_LAYERS} layers.")

    # 1. Create the VQE circuit
    vqe_circuit, hamiltonian = create_vqe_circuit(N_LAYERS, N_QUBITS)
    print("\nExact ground state energy (for reference):")
    print(qml.eigvals(hamiltonian)[0])


    # 2. Generate random parameter sets to sample the landscape
    param_samples = generate_parameter_samples(N_LAYERS, N_QUBITS, N_SAMPLES)

    # 3. Calculate the maximum Hessian norm (L_max) from the samples
    L_max = calculate_max_hessian_norm(vqe_circuit, param_samples)

    # 4. Calculate the optimal learning rate based on the heuristic η ≈ 1/L
    eta_optimal = 1.0 / L_max

    print("\n--- Results ---")
    print(f"Estimated Maximum Curvature (L_max): {L_max:.4f}")
    print(f"Calculated Optimal Learning Rate (η ≈ 1/L_max): {eta_optimal:.4f}")
    print("\nUse this optimal learning rate in the 'train_vqe_demo.py' script.")