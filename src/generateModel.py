import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from pennylane import numpy as qml_np
import pennylane as qml
import tensorflow as tf
from tensorflow import keras
import joblib
from saveModel import saveRNNModel

# Function to clean data
def clean_data(data):
    redundant_states = [
        'Screen off (locked)', 'Screen on (unlocked)', 'Screen off (unlocked)', 
        'Screen on (locked)', 'Screen on', 'Screen off', 'Device shutdown', 'Device boot'
    ]
    for state in redundant_states:
        data = data[~(data == state).any(axis=1)]
    data = data.dropna()
    data.index = range(len(data))
    return data

# Function to encode data
def encode_data(data):
    label_encoder_app = LabelEncoder()
    encoded_data = label_encoder_app.fit_transform(data.iloc[:, 0])
    encoded_data = pd.DataFrame(data=encoded_data)
    return encoded_data, label_encoder_app

# Function to split data into training and testing sets
def split_into_train_test_set(encoded_data):
    train_set = encoded_data.iloc[:1901].values
    test_set = encoded_data.iloc[1901:].values
    return train_set, test_set

# Function to create the quantum LSTM model
def create_quantum_lstm_model(num_qubits):
    num_layers = 1

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface='autograd')
    def quantum_lstm_layer(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]  # Output should match the number of qubits

    weight_shapes = {"weights": (num_layers, num_qubits)}
    qlayer = qml.qnn.KerasLayer(quantum_lstm_layer, weight_shapes, output_dim=num_qubits)  # Match output_dim with num_qubits
    return qlayer

# Cost function for quantum LSTM
def cost_fn(X_train, y_train, qlayer, dense_layer):
    predictions = dense_layer(qlayer(X_train))
    return np.mean((predictions - y_train) ** 2)

# Main script
use_preTrained_model = False
if len(sys.argv) > 1:
    use_preTrained_model = sys.argv[1]

data = pd.read_csv('Datasets/dataset.csv')
data = clean_data(data)
encoded_data, label_encoder_app = encode_data(data)
train_set, test_set = split_into_train_test_set(encoded_data)

scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(train_set)

X_train = []
y_train = []

for i in range(10, 1901):
    X_train.append(training_set_scaled[i-10:i, 0])
    y_train.append(train_set[i, 0])

X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])

label_encoder_y = LabelEncoder()
y_train = label_encoder_y.fit_transform(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes=36)

# Use PCA to reduce the feature dimension to the number of qubits
num_qubits = 10
pca = PCA(n_components=num_qubits)
X_train_pca = pca.fit_transform(X_train)

# Create the quantum LSTM model
qlayer = create_quantum_lstm_model(num_qubits)

# Add a dense layer to match the output shape with y_train
dense_layer = keras.layers.Dense(36, activation='softmax')

# Initialize the weights
weights = qml_np.random.random((1, num_qubits))

# Optimizer
opt = qml.GradientDescentOptimizer(0.01)

# Training loop
for epoch in range(100):  # 100 epochs
    weights = opt.step(lambda w: cost_fn(X_train_pca, y_train, qlayer, dense_layer), weights)
    cost = cost_fn(X_train_pca, y_train, qlayer, dense_layer)
    print(f"Epoch {epoch + 1}: Cost = {cost}")

# Testing the model
total_dataset = encoded_data.iloc[:, 0]
inputs = total_dataset[len(total_dataset) - len(test_set) - 10:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []

for i in range(10, 397):
    X_test.append(inputs[i-10:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Apply PCA to the test data
X_test_pca = pca.transform(X_test)

# Make predictions
predicted_app = dense_layer(qlayer(X_test_pca))
predicted_app = predicted_app.numpy()  # Convert EagerTensor to NumPy array
predicted_app_1 = np.argmax(predicted_app, axis=1)

# Confusion matrix (for checking accuracy)
cm = np.zeros(shape=(2, 2))
for i in range(387):
    if test_set[i] == predicted_app_1[i]:
        cm[1, 1] += 1
    else:
        cm[1, 0] += 1

# Indices of the highest values
idx = (-predicted_app).argsort(axis=1)

# Prediction and actual apps used
prediction = label_encoder_app.inverse_transform(idx[:, 0])
prediction = pd.DataFrame(data=prediction)
actual_app_used = label_encoder_app.inverse_transform(test_set)
actual_app_used = pd.DataFrame(data=actual_app_used)

for i in range(1, 4):
    idx_i = label_encoder_app.inverse_transform(idx[:, i])
    idx_i = pd.DataFrame(data=idx_i)
    prediction = pd.concat([prediction, idx_i], axis=1)

# Combine predictions and actual values
final_outcome = pd.concat([prediction, actual_app_used], axis=1)
final_outcome.columns = ['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4', 'Actual App Used']
print('***********************************FINAL PREDICTION*********************************')
print(final_outcome)

# Saving the model
if use_preTrained_model != 'True':
    saveRNNModel(qlayer)
