import pennylane as qml
import pennylane.numpy as pnp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd

# 2 layers, 4 qubits
two_four = pnp.array([
    [[1.74893948, 4.97321702],
     [5.1100455, 0.2901696],
     [5.84932921, 5.95260408],
     [5.17407071, 4.25346973]],

    [[3.38418483, 0.12883222],
     [2.35666436, 1.02604641],
     [1.14358045, 3.23624769],
     [5.68954706, 2.53291517]]
])

# 3 layers, 8 qubits
three_eight = pnp.array([
    [[0.41657189, 6.06150163],
     [4.73310445, 4.0003747],
     [1.5830972, 2.40750344],
     [0.18034015, 2.36990932],
     [4.77822052, 3.67074511],
     [5.215826, 0.32375785],
     [2.47905865, 2.22915034],
     [4.23111491, 0.58666382]],

    [[2.9492437, 1.2216882],
     [5.33997835, 0.23517074],
     [0.45570561, 2.78765853],
     [0.55174352, 1.6204769],
     [4.18441315, 5.67103004],
     [3.39021748, 5.1679197],
     [0.19996743, 5.18508334],
     [5.59398687, 3.36943771]],

    [[6.27613973, 4.87414451],
     [3.25158254, 1.53296903],
     [5.70430834, 4.88991285],
     [3.65215324, 2.97141491],
     [6.03598432, 2.54583038],
     [3.38965927, 4.10623231],
     [5.02961429, 4.92675931],
     [1.59282892, 4.13939312]]])

# 5 layers, 10 qubits
five_ten = pnp.array([
    [[4.9158418, 2.7880879],
     [6.02319736, 3.66258225],
     [3.25762903, 4.84276927],
     [4.07924941, 2.0868001],
     [4.88622513, 0.0127704],
     [0.6708496, 1.41914373],
     [0.04638703, 1.46583988],
     [0.47877635, 2.84645249],
     [3.92576205, 4.44752135],
     [2.67823419, 4.81162046]],

    [[0.69093245, 1.39371023],
     [5.97960255, 1.29464209],
     [5.32581545, 0.92807303],
     [0.62192281, 4.57534213],
     [0.11102478, 5.09433613],
     [0.53844149, 3.71189042],
     [4.52150829, 0.34344321],
     [4.19577808, 3.99700366],
     [5.80927188, 5.88799466],
     [4.37510861, 3.28794424]],

    [[2.65869432, 1.96148726],
     [2.08538198, 1.59790511],
     [1.36881248, 2.1268339],
     [2.162733, 2.99924455],
     [5.60542736, 1.59950401],
     [0.44802593, 3.61089321],
     [6.16621735, 5.85325037],
     [1.2819916, 3.9476495],
     [4.9225963, 4.6628248],
     [4.08751644, 4.50646831]],

    [[2.3965714, 2.6047425],
     [3.72446594, 0.37937507],
     [1.4490902, 0.922114],
     [2.22214321, 0.4365484],
     [1.97901594, 6.25975931],
     [3.71365806, 2.49114045],
     [0.86973788, 0.36317717],
     [1.00657735, 3.69862089],
     [2.5058345, 3.57435732],
     [6.00939383, 3.74259911]],

    [[4.73107702, 3.47510882],
     [1.99680147, 2.50855254],
     [2.47956684, 1.13589699],
     [0.60339085, 3.80312564],
     [0.16736191, 2.5796839],
     [5.27447934, 1.72167448],
     [0.14467851, 4.22226496],
     [5.34522621, 4.74136408],
     [0.70798205, 1.71333788],
     [0.36424083, 0.20817739]]])

def create_qnn(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, params):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        for layer in range(n_layers):
            # Rotational Gates
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RZ(params[layer][qubit][1], wires=qubit)

            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CZ(wires=[qubit, next_qubit])

        return qml.probs(wires=range(2))

    return circuit

def create_qnn_non_smooth(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    def triangle_activation(x):
        """ A periodic, non-smooth activation function for angles. """
        # Scale x to be in a period of 2*pi, then apply a triangle wave
        # The core of the non-smoothness comes from pnp.abs()
        return pnp.abs((x % (2 * pnp.pi)) - pnp.pi)

    @qml.qnode(dev)
    def circuit(inputs, params):
        # Data Encoding
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        # Variational Layers
        for layer in range(n_layers):
            # Rotational Gates with non-smooth activation
            for qubit in range(n_qubits):
                # Apply the activation function to the parameter before using it.
                # The gradient of this function is a step function, which makes
                # the overall loss landscape non-smooth.
                activated_param_rx = triangle_activation(params[layer][qubit][0])
                activated_param_rz = triangle_activation(params[layer][qubit][1])

                qml.RX(activated_param_rx, wires=qubit)
                qml.RZ(activated_param_rz, wires=qubit)

            # Entangling Gates
            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CZ(wires=[qubit, next_qubit])

        return qml.probs(wires=range(2))

    return circuit

def cross_entropy_loss(output_distribution, true):
    epsilon = 1e-10
    pred = output_distribution[true]
    return -pnp.log(pred + epsilon)

def preprocess_image(x, n_components):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # reduce dimension
    x_scaled = scaler.fit_transform(x)
    x_pca = pca.fit_transform(x_scaled)

    # Normalize to [0, 2π] for angle encoding
    x_pca_normalized = 2.0 * pnp.pi * (x_pca - x_pca.min(axis=0)) / (x_pca.max(axis=0) - x_pca.min(axis=0))

    return x_pca_normalized

def train_qnn_param_shift(x, y, n_layers, n_qubits, n_gates, n_epochs, lr=0.001, smooth=True):
    if smooth:
        forward_pass = create_qnn(n_layers, n_qubits)
        print("Using a Smooth QNN")
    else:
        forward_pass = create_qnn_non_smooth(n_layers, n_qubits)
        print("Non-Smooth QNN")

    fp = 0
    params = three_eight
    loss_history = []
    fp_history = []

    # Cross entropy loss function
    def cost_fn(params, image, label):
        pred = forward_pass(image, params)[label]
        return -pnp.log(pred + 1e-10)

    grad_fn = qml.grad(cost_fn, argnum=0)

    """Training Loop"""
    for time_step in tqdm(range(n_epochs), desc="Time step"):
        s = 100
        x_t = x[time_step * s:(time_step + 1) * s]
        y_t = y[time_step * s:(time_step + 1) * s]
        epoch_loss = 0
        correct_predictions = 0

        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {time_step + 1}/{n_epochs}", leave=False):
            # Compute loss with current parameters
            out = forward_pass(image, params)
            fp += 1
            loss = cross_entropy_loss(out, label)
            epoch_loss += loss

            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1

            # compute gradients and apply only to active params
            gradients = grad_fn(params, image, label)

            # increase fp by 2*n_active_params
            fp += 2 * params.size

            # Update active params only
            params -= lr * gradients

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {time_step + 1}/{n_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history


# def calculate_hessian_norms(qnn, samples):
#     hessian_norms = []
#     hessian_fn = qml.jacobian(qml.grad(qnn))
#     for i, params in enumerate(samples):
#         # Output of QNN only accepts flat parameter vector
#         flat_params = params.flatten()
#         def cost_fn_flat(p_flat):
#             p_reshaped = p_flat.reshape(params.shape)
#             return qnn(p_reshaped)
#
#         # Calculate hessian matrix
#         hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)
#
#         # Calculate the spectral norm (largest singular value) of the Hessian = largest absolute eigenvalue.
#         spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
#         hessian_norms.append(spectral_norm)
#
#     return hessian_norms


def calculate_hessian_norms(qnn, parameter_sets, data_x, data_y):
    """
    Calculates the spectral norm of the Hessian of the loss function.

    Args:
        qnn (qml.QNode): The quantum neural network circuit.
        parameter_sets (pnp.array): An array of parameter sets to evaluate.
                                    The function will calculate a Hessian for each set.
        data_x (pnp.array): The input feature data.
        data_y (pnp.array): The input label data.
    """
    hessian_norms = []

    # We will calculate the Hessian with respect to the first data sample.
    # In a more advanced scenario, you might average Hessians over a batch.
    image = data_x[0]
    label = data_y[0]

    print(f"Calculating Hessian norms for {len(parameter_sets)} parameter set(s)...")
    print(f"Using data point 0 with label {label} for loss calculation.")

    for params in tqdm(parameter_sets, desc="Processing Parameter Sets"):
        # The original shape is needed to reshape the flat vector inside the cost function.
        original_shape = params.shape

        # Define a cost function that takes a flat parameter vector,
        # reshapes it, and computes the loss for our chosen data point.
        def cost_fn_flat(p_flat):
            # Reshape parameters to the format expected by the QNN
            p_reshaped = p_flat.reshape(original_shape)

            # Get the QNN output distribution for the specific image
            output_distribution = qnn(image, p_reshaped)

            # Calculate the cross-entropy loss
            return cross_entropy_loss(output_distribution, label)

        # Flatten the current parameter set to pass to the new cost function
        flat_params = params.flatten()

        # Calculate the Hessian matrix (Jacobian of the gradient)
        hessian_matrix = qml.jacobian(qml.grad(cost_fn_flat))(flat_params)

        # Calculate the spectral norm (largest singular value), which for a
        # symmetric matrix like the Hessian is the largest absolute eigenvalue.
        spectral_norm = pnp.linalg.norm(hessian_matrix, ord=2)
        hessian_norms.append(spectral_norm)

    return hessian_norms

# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

n_samples = 100
n_qubits = 4
n_layers = 2
n_gates = 2
n_epochs = 300
x = preprocess_image(x, n_qubits)

print(f"{n_layers} Layers, {n_qubits} Qubits")
#train_qnn_param_shift(x, y, n_layers, n_qubits, n_gates, n_epochs, lr=0.01, smooth=True)
my_qnn = create_qnn(n_layers, n_qubits)
res = calculate_hessian_norms(my_qnn, two_four, x, y)
print(res)