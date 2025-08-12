import pennylane as qml
import pennylane.numpy as pnp
from qiskit.providers.fake_provider import Fake127QPulseV1
from qiskit_aer.noise import NoiseModel

def get_noisy_device(n_qubits):
    # 1. Get the fake backend object. This is an offline snapshot of a real IBMQ device.
    fake_backend = Fake127QPulseV1()
    print(f"Using noise model from: {fake_backend.name}")

    # 2. Create a noise model from the backend's properties.
    noise_model = NoiseModel.from_backend(fake_backend)

    # 3. Create a PennyLane device using qiskit.aer, a high-performance Qiskit simulator.
    dev = qml.device(
        'qiskit.aer',
        wires=n_qubits,
        noise_model=noise_model,
        shots=1000
    )
    print("Device created successfully.")
    return dev

def create_qnn(dev, n_layers, n_qubits, n_gates):
    @qml.qnode(dev, diff_method="finite-diff")
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

def generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=2):
    """
    generates n_samples random parameter tensors
    """
    pnp.random.seed(42)
    samples = [pnp.random.uniform(0, 2*pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]
    return pnp.array(samples)

def calculate_hessian_norms(qnn, samples):
    hessian_norms = []
    for i, params in enumerate(samples):
        print(f"Calculating Hessian for sample {i + 1}/{len(samples)}...")
        flat_params = params.flatten()
        def cost_fn_flat(p_flat):
            p_reshaped = p_flat.reshape(params.shape)
            return qnn(p_reshaped)

        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        hessian_norms.append(spectral_norm)

    return hessian_norms


# ==================================================================
if __name__ == '__main__':
    # --- Configuration ---
    n_qubit_combos = [1, 2, 4, 8]
    n_layers = 1
    n_gates = 1
    n_samples = 10

    for n_qubits in n_qubit_combos:
        # --- Device and QNN Creation ---
        # 1. Get the noisy device
        noisy_device = get_noisy_device(n_qubits)

        # 2. Create the QNN, passing the noisy device to it
        qnn = create_qnn(noisy_device, n_layers, n_qubits, n_gates)

        # --- Data Generation ---
        samples = generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=n_gates)

        # --- Theoretical Bound Calculation ---
        P = n_layers * n_qubits * n_gates
        norm_M = 1.0
        L_bound = P * norm_M

        # --- Experiment ---
        hessian_norms = calculate_hessian_norms(qnn, samples)

        # --- Verification & Output ---
        print("--- Experiment Setup ---")
        print(f"Number of Qubits: {n_qubits}, Number of Layers: {n_layers}, Total Parameters (P): {P}")
        print(f"Theoretical L-Smoothness Bound (L <= P): {L_bound:.4f}")
        print(f"Largest Hessian Norm: {pnp.max(hessian_norms)}")