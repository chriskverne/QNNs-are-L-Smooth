import pennylane as qml
from pennylane import numpy as pnp
import itertools

# --- Configuration for 1-Qubit Experiment ---
N_QUBITS = 1
N_LAYERS = 40 # A deep circuit to create a high-P landscape
N_GATES_PER_ROTATION = 3 # RX, RY, RZ
N_SAMPLES = 200

def create_vqe_circuit(n_layers, n_qubits):
    """
    Creates the VQE circuit for a SINGLE qubit.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    # A simple 1-qubit Hamiltonian (e.g., spin in a magnetic field)
    coeffs = [0.6, 0.8]
    obs = [qml.PauliX(0), qml.PauliZ(0)]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev)
    def circuit(params):
        """
        Ansatz for the VQE. Note: No entanglement for n=1.
        """
        for layer in range(n_layers):
            # Layer of rotational gates
            for qubit in range(n_qubits): # This loop will only run for qubit 0
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RY(params[layer][qubit][1], wires=qubit)
                qml.RZ(params[layer][qubit][2], wires=qubit)

        return qml.expval(hamiltonian)

    return circuit, hamiltonian


def generate_parameter_samples(n_layers, n_qubits, n_samples):
    """
    Generates n_samples random parameter tensors for exploring the landscape.
    """
    pnp.random.seed(42)
    param_shape = (n_layers, n_qubits, N_GATES_PER_ROTATION)
    samples = [pnp.random.uniform(0, 2 * pnp.pi, size=param_shape) for _ in range(n_samples)]
    return pnp.array(samples)


def calculate_max_hessian_norm(qnode, samples):
    """
    Calculates the spectral norm of the Hessian for each parameter sample
    and returns the maximum observed value (our estimate for L_max).
    """
    hessian_norms = []
    param_shape = samples[0].shape

    print(f"Starting Hessian norm calculation for {len(samples)} samples...")
    for i, params in enumerate(samples):
        flat_params = params.flatten()
        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(param_shape)
            return qnode(p_reshaped)

        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        hessian_norms.append(spectral_norm)

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i+1}/{len(samples)} samples.")

    return pnp.max(hessian_norms)


if __name__ == '__main__':
    print("--- 1-Qubit VQE Landscape Analysis ---")
    print(f"Circuit details: {N_QUBITS} qubit, {N_LAYERS} layers.")

    vqe_circuit, hamiltonian = create_vqe_circuit(N_LAYERS, N_QUBITS)
    print("\nExact ground state energy (for reference):")
    # For H = aX + cZ, the eigenvalues are +/- sqrt(a^2 + c^2)
    coeffs = [0.6, 0.8]
    exact_energy = -pnp.sqrt(coeffs[0]**2 + coeffs[1]**2)
    print(exact_energy)

    param_samples = generate_parameter_samples(N_LAYERS, N_QUBITS, N_SAMPLES)
    L_max = calculate_max_hessian_norm(vqe_circuit, param_samples)
    eta_optimal = 1.0 / L_max

    print("\n--- Results ---")
    print(f"Estimated Maximum Curvature (L_max): {L_max:.4f}")
    print(f"Calculated Optimal Learning Rate (η ≈ 1/L_max): {eta_optimal:.4f}")

