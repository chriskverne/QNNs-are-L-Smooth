# import pennylane as qml
# import pennylane.numpy as pnp
# import matplotlib.pyplot as plt
#
#
# def create_qnn_custom(n_layers, n_qubits, generators, observable_coeffs, observable_ops):
#     """
#     Creates a QNN with custom, user-defined generators by mapping them
#     to equivalent, numerically stable standard Pauli rotations.
#     """
#     dev = qml.device('default.qubit', wires=n_qubits)
#
#     @qml.qnode(dev)
#     def circuit(params):
#         for l in range(n_layers):
#             for q in range(n_qubits):
#                 for g in range(len(generators[l][q])):
#                     generator_op = generators[l][q][g]
#                     param = params[l][q][g]
#
#                     coeff = generator_op.scalar
#                     base_op = generator_op.base
#                     wires = base_op.wires
#
#                     if isinstance(base_op, qml.PauliX):
#                         qml.RX(-2 * coeff * param, wires=wires)
#                     elif isinstance(base_op, qml.PauliY):
#                         qml.RY(-2 * coeff * param, wires=wires)
#                     elif isinstance(base_op, qml.PauliZ):
#                         qml.RZ(-2 * coeff * param, wires=wires)
#
#         observable = qml.Hamiltonian(observable_coeffs, observable_ops)
#         return qml.expval(observable)
#
#     return circuit
#
#
# def generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates):
#     pnp.random.seed(42)
#     return [pnp.random.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]
#
#
# def calculate_max_hessian_norm(qnn, samples):
#     max_norm = 0
#     for i, params in enumerate(samples):
#         flat_params = params.flatten()
#
#         def cost_fn_flat(p_flat):
#             return qnn(p_flat.reshape(params.shape))
#
#         hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
#         spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
#         if spectral_norm > max_norm:
#             max_norm = spectral_norm
#     return max_norm
#
#
# # ==================================================================
# # Main execution block for Heterogeneity Sweep Experiment
# # ==================================================================
# if __name__ == '__main__':
#     # --- Experiment Constants ---
#     N_QUBITS = 4
#     N_LAYERS = 5
#     N_GATES = 3
#     N_SAMPLES = 10
#     P = N_QUBITS * N_LAYERS * N_GATES
#
#     # --- Define Generator Templates ---
#     weak_templates = [(0.5, qml.PauliX), (0.5, qml.PauliY), (0.5, qml.PauliZ)]
#     strong_templates = [(1.5, qml.PauliX), (1.5, qml.PauliY), (1.5, qml.PauliZ)]
#
#     # --- The Sweep ---
#     # We sweep the percentage of "strong" generator sets in the circuit
#     mix_ratios = pnp.linspace(0, 1, 11)  # 0%, 10%, 20%, ..., 100%
#
#     results = []
#
#     print("--- Running Generator Heterogeneity Sweep ---")
#     print(f"Constant circuit size: {N_QUBITS} Qubits, {N_LAYERS} Layers, P={P} Params")
#
#     for mix_ratio in mix_ratios:
#         print(f"\nProcessing mix ratio: {mix_ratio * 100:.0f}% strong generators...")
#
#         # --- Build the specific generator set for this mix ratio ---
#         num_locations = N_QUBITS * N_LAYERS
#         # Convert PennyLane tensor to a standard float before rounding
#         num_strong = int(round(float(mix_ratio) * num_locations))
#
#         # Create a shuffled list of location types ('strong' or 'weak') for reproducibility
#         location_types = ['strong'] * num_strong + ['weak'] * (num_locations - num_strong)
#         pnp.random.default_rng(seed=42).shuffle(location_types)
#
#         circuit_generators = []
#         loc_idx = 0
#         for l in range(N_LAYERS):
#             layer_gens = []
#             for q in range(N_QUBITS):
#                 if location_types[loc_idx] == 'strong':
#                     templates = strong_templates
#                 else:
#                     templates = weak_templates
#                 layer_gens.append([coeff * op(q) for coeff, op in templates])
#                 loc_idx += 1
#             circuit_generators.append(layer_gens)
#
#         # --- QNN and Data Generation ---
#         obs_coeffs = [1 / N_QUBITS] * N_QUBITS
#         obs_ops = [qml.PauliZ(i) for i in range(N_QUBITS)]
#         qnn = create_qnn_custom(N_LAYERS, N_QUBITS, circuit_generators, obs_coeffs, obs_ops)
#         samples = generate_parameter_samples(N_LAYERS, N_QUBITS, N_SAMPLES, N_GATES)
#
#         # --- Theoretical Bound Calculation ---
#         norm_M = 1.0
#         flat_generators = [g for layer in circuit_generators for qubit in layer for g in qubit]
#         generator_norms = [abs(g.scalar) for g in flat_generators]
#
#         our_bound = 4 * norm_M * sum(n ** 2 for n in generator_norms)
#         liu_bound = 4 * P * norm_M * max(n ** 2 for n in generator_norms)
#         tightness_ratio = liu_bound / our_bound if our_bound > 0 else 1.0
#
#         # --- Run Simulation (optional, can be commented out if only bounds are needed) ---
#         # max_measured_norm = calculate_max_hessian_norm(qnn, samples)
#
#         print(f"Our Bound: {our_bound:.2f} | Liu Bound: {liu_bound:.2f} | Ratio (Liu/Our): {tightness_ratio:.3f}")
#
#         results.append({
#             'mix_ratio': mix_ratio,
#             'our_bound': our_bound,
#             'liu_bound': liu_bound,
#             'tightness_ratio': tightness_ratio
#         })
#
#     # --- Print Final Results Table ---
#     print("\n\n" + "=" * 60)
#     print("                 FINAL RESULTS SUMMARY")
#     print("=" * 60)
#     print(f"{'Mix Ratio (%)':>15} | {'Our Bound':>12} | {'Liu Bound':>12} | {'Tightness Ratio':>15}")
#     print("-" * 60)
#     for res in results:
#         print(
#             f"{res['mix_ratio'] * 100:>14.0f}% | {res['our_bound']:>12.2f} | {res['liu_bound']:>12.2f} | {res['tightness_ratio']:>15.3f}")
#     print("=" * 60)
#


import pennylane as qml
import pennylane.numpy as pnp
import matplotlib.pyplot as plt


def create_qnn_custom(n_layers, n_qubits, generators, observable_coeffs, observable_ops):
    """
    Creates a QNN with custom, user-defined generators by mapping them
    to equivalent, numerically stable standard Pauli rotations.
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for l in range(n_layers):
            for q in range(n_qubits):
                for g in range(len(generators[l][q])):
                    generator_op = generators[l][q][g]
                    param = params[l][q][g]

                    coeff = generator_op.scalar
                    base_op = generator_op.base
                    wires = base_op.wires

                    if isinstance(base_op, qml.PauliX):
                        qml.RX(-2 * coeff * param, wires=wires)
                    elif isinstance(base_op, qml.PauliY):
                        qml.RY(-2 * coeff * param, wires=wires)
                    elif isinstance(base_op, qml.PauliZ):
                        qml.RZ(-2 * coeff * param, wires=wires)

        observable = qml.Hamiltonian(observable_coeffs, observable_ops)
        return qml.expval(observable)

    return circuit


def generate_parameter_samples(n_layers, n_qubits, n_samples, n_gates):
    pnp.random.seed(42)
    return [pnp.random.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits, n_gates)) for _ in range(n_samples)]


def calculate_max_hessian_norm(qnn, samples):
    max_norm = 0
    print("Running Hessian calculations for samples:")
    for i, params in enumerate(samples):
        print(f"  - Sample {i + 1}/{len(samples)}...", end='\r')
        flat_params = params.flatten()

        def cost_fn_flat(p_flat):
            return qnn(p_flat.reshape(params.shape))

        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        if spectral_norm > max_norm:
            max_norm = spectral_norm
    print(f"\n  ...Done. Max norm found: {max_norm:.4f}")
    return max_norm


# ==================================================================
# Main execution block for Heterogeneity Sweep Experiment
# ==================================================================
if __name__ == '__main__':
    # --- Experiment Constants ---
    N_QUBITS = 4
    N_LAYERS = 5
    N_GATES = 3
    N_SAMPLES = 100
    P = N_QUBITS * N_LAYERS * N_GATES

    # --- Define Generator Templates ---
    weak_templates = [(0.5, qml.PauliX), (0.5, qml.PauliY), (0.5, qml.PauliZ)]
    strong_templates = [(1.5, qml.PauliX), (1.5, qml.PauliY), (1.5, qml.PauliZ)]

    # --- The Sweep ---
    # We sweep the percentage of "strong" generator sets in the circuit
    mix_ratios = pnp.linspace(0, 1, 11)  # 0%, 10%, 20%, ..., 100%

    results = []

    print("--- Running Generator Heterogeneity Sweep ---")
    print(f"Constant circuit size: {N_QUBITS} Qubits, {N_LAYERS} Layers, P={P} Params")

    for mix_ratio in mix_ratios:
        print(f"\nProcessing mix ratio: {mix_ratio * 100:.0f}% strong generators...")

        # --- Build the specific generator set for this mix ratio ---
        num_locations = N_QUBITS * N_LAYERS
        num_strong = int(round(float(mix_ratio) * num_locations))

        location_types = ['strong'] * num_strong + ['weak'] * (num_locations - num_strong)
        pnp.random.default_rng(seed=42).shuffle(location_types)

        circuit_generators = []
        loc_idx = 0
        for l in range(N_LAYERS):
            layer_gens = []
            for q in range(N_QUBITS):
                templates = strong_templates if location_types[loc_idx] == 'strong' else weak_templates
                layer_gens.append([coeff * op(q) for coeff, op in templates])
                loc_idx += 1
            circuit_generators.append(layer_gens)

        # --- QNN and Data Generation ---
        obs_coeffs = [1 / N_QUBITS] * N_QUBITS
        obs_ops = [qml.PauliZ(i) for i in range(N_QUBITS)]
        qnn = create_qnn_custom(N_LAYERS, N_QUBITS, circuit_generators, obs_coeffs, obs_ops)
        samples = generate_parameter_samples(N_LAYERS, N_QUBITS, N_SAMPLES, N_GATES)

        # --- Theoretical Bound Calculation ---
        norm_M = 1.0
        flat_generators = [g for layer in circuit_generators for qubit in layer for g in qubit]
        generator_norms = [abs(g.scalar) for g in flat_generators]

        our_bound = 4 * norm_M * sum(n ** 2 for n in generator_norms)
        liu_bound = 4 * P * norm_M * max(n ** 2 for n in generator_norms)
        tightness_ratio = liu_bound / our_bound if our_bound > 0 else 1.0

        # --- Run Full Simulation ---
        max_measured_norm = calculate_max_hessian_norm(qnn, samples)
        empirical_ratio = max_measured_norm / our_bound  if max_measured_norm > 0 else float('inf')

        results.append({
            'mix_ratio': mix_ratio,
            'our_bound': our_bound,
            'liu_bound': liu_bound,
            'tightness_ratio': tightness_ratio,
            'max_measured_norm': max_measured_norm,
            'empirical_ratio': empirical_ratio
        })

    # --- Print Final Results Table ---
    print("\n\n" + "=" * 85)
    print("                                   FINAL RESULTS SUMMARY")
    print("=" * 85)
    header = f"{'Mix Ratio (%)':>15} | {'Measured Lmax':>15} | {'Our Bound':>12} | {'Liu Bound':>12} | {'Theory Ratio':>15} | {'Empirical Ratio':>15}"
    print(header)
    print("-" * 85)
    for res in results:
        line = f"{res['mix_ratio'] * 100:>14.0f}% | {res['max_measured_norm']:>15.4f} | {res['our_bound']:>12.2f} | {res['liu_bound']:>12.2f} | {res['tightness_ratio']:>15.3f} | {res['empirical_ratio']:>15.2f}"
        print(line)
    print("=" * 85)
    print("* Theory Ratio = Liu Bound / Our Bound  (Higher is better for us)")
    print("* Empirical Ratio = Our Bound / Measured Lmax (Measures how tight our bound is to reality)")

