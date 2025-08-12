import pennylane as qml
import pennylane.numpy as pnp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd

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

def train_qnn_param_shift(x, y, n_layers, n_qubits, n_gates, n_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    fp = 0
    params = pnp.random.uniform(0, 2*pnp.pi, size=(n_layers, n_qubits, n_gates))
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
            params -= 0.001 * gradients


        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {time_step + 1}/{n_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history


# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('./data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

n_qubits = 8
n_layers = 3
n_gates = 2
n_epochs = 300
x = preprocess_image(x, n_qubits)

train_qnn_param_shift(x, y, n_qubits, n_layers, n_gates, n_epochs)