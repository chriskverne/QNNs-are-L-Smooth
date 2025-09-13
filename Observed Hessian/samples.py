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
    #pnp.random.seed(42)
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
    results_data = []

    """
    Experiment 1: Fix N_qubits, N_Gates, Increase N_Layers
    Either 2,3 or 4 qubits sounds good
    N_Gates = 2 or 3 sounds good
    """
    n_samples = 1000
    n_gates = 3
    n_qubits = 4
    n_layers = 4 #[1,5,10,15,20,25,30,35,40]#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    entanglement = False
    observable_coeffs = [1 / n_qubits] * n_qubits # STANDARD Zi MEASURMENT ALL QUBITS
    observable_ops = [qml.PauliZ(i) for i in range(n_qubits)] # STANDARD Zi MEASURMENT ALL QUBITS

    # --- QNN and Data Generation ---
    qnn = create_qnn(n_layers, n_qubits, n_gates, observable_coeffs, observable_ops, entangled=entanglement)
    samples = generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates=n_gates)

    # --- Theoretical Bound Calculation ---
    P = n_layers * n_qubits * n_gates
    norm_M = 1.0
    L_bound = P * norm_M

    # --- Experiment ---
    hessian_norms = calculate_hessian_norms(qnn, samples)

    # --- Verification ---
    all_within_bound = all(norm <= L_bound for norm in hessian_norms)
    print("--- Experiment Setup ---")
    print(f"Number of Layers: {n_layers}, Number of Qubits: {n_qubits}, Number of Gates: {n_gates}, Total Parameters (P): {P}")
    print(f"Theoretical L-Smoothness Bound (L <= P): {L_bound:.4f}")
    print(f"Largest Hessian Norm: {pnp.max(hessian_norms)}")
    #results_data.append((n_layers, n_qubits, n_gates, pnp.max(hessian_norms)))
    print(hessian_norms)

    # r1 =  hessian_norms[0:10] #10 first samples
    # r2 = hessian_norms[10:60] #50 next samples
    # r3 = hessian_norms[60:160] #100
    # r4 = hessian_norms[160:410] #250
    # r5 = hessian_norms[410:910] #500
    # r6 = hessian_norms[910:1910] #1000
    # r7 = hessian_norms[1910:3910] #2000
    # r8 = hessian_norms[3910:7910] #4000
    #print('10 samples: ',pnp.max(r1), ' 50 samples: ', pnp.max(r2), ' 100 samples: ', pnp.max(r3), ' 250 samples: ', pnp.max(r4),' 500 samples: ', pnp.max(r5), ' 1000 samples: ',pnp.max(r6), ' 2000 samples: ',pnp.max(r7), '4000 samples: ',pnp.max(r8))
    print(results_data)
    ######################
    # 10
    # samples: 2.6805072639030025
    # 50
    # samples: 2.5269121135139216
    # 100
    # samples: 2.8551045049470862
    # 250
    # samples: 2.910669709448535
    # 500
    # samples: 2.9079474983639226
    # 1000
    # samples: 2.9173548644182864
    # 2000
    # samples: 2.968031440279984
    # 4000
    # samples: 2.9363644775244446