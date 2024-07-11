import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from pennylane import numpy as pnp

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

# Create a PennyLane quantum device
dev = qml.device("default.qubit", wires=4)

# Define a quantum layer
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(w)) for w in range(4)]

weight_shapes = {"weights": (2, 4, 3)}

# Create the QNode layer
class QNodeLayer(nn.Module):
    def __init__(self):
        super(QNodeLayer, self).__init__()
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        x = self.qlayer(x)
        return x

# Define the Quantum LSTM cell
class QuantumLSTMCell(nn.Module):
    def __init__(self):
        super(QuantumLSTMCell, self).__init__()
        self.dense = nn.Linear(4, 4)
        self.qnode_layer = QNodeLayer()

    def forward(self, x, hidden):
        combined = x + hidden
        combined = self.dense(combined)
        output = self.qnode_layer(combined)
        return output, output

# Define the Quantum LSTM layer
class QuantumLSTMLayer(nn.Module):
    def __init__(self, input_dim):
        super(QuantumLSTMLayer, self).__init__()
        self.cell = QuantumLSTMCell()
        self.hidden_size = input_dim

    def forward(self, x):
        outputs = []
        h_t = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        for t in range(x.size(1)):
            h_t, _ = self.cell(x[:, t, :], h_t)
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)

# Define the quantum LSTM model
class QuantumLSTMModel(nn.Module):
    def __init__(self):
        super(QuantumLSTMModel, self).__init__()
        self.qlstm1 = QuantumLSTMLayer(input_dim=4)
        self.dropout1 = nn.Dropout(0.2)
        self.qlstm2 = QuantumLSTMLayer(input_dim=4)
        self.dropout2 = nn.Dropout(0.2)
        self.dense = nn.Linear(4, 36)

    def forward(self, x):
        x = self.qlstm1(x)
        x = self.dropout1(x)
        x = self.qlstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.dense(x)
        return x

# Main script
data = pd.read_csv('/content/dataset.csv')
data = clean_data(data)
encoded_data, label_encoder_app = encode_data(data)
train_set, test_set = encoded_data.iloc[:1901].values, encoded_data.iloc[1901:].values

scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(train_set)

X_train = []
y_train = []

for i in range(10, 1901):
    X_train.append(training_set_scaled[i-10:i, 0])
    y_train.append(train_set[i, 0])

X_train = np.array(X_train)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Ensure the tensor has 3 dimensions (batch_size, seq_len, feature_dim)

label_encoder_y = LabelEncoder()
y_train = label_encoder_y.fit_transform(y_train)
y_train = torch.tensor(y_train, dtype=torch.long)

# Create data loaders
batch_size = 32
train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# Split the training data into training and validation sets
train_size = int(0.8 * len(train_data.dataset))
val_size = len(train_data.dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data.dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Compile and train the model
model = QuantumLSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {accuracy}%")

# Save the model
torch.save(model.state_dict(), 'quantum_lstm_model.pth')

# Testing the model
X_test = []
inputs = training_set_scaled[len(training_set_scaled) - len(test_set) - 10:].reshape(-1, 1)
inputs = scaler.transform(inputs)

for i in range(10, 397):
    X_test.append(inputs[i-10:i, 0])

X_test = np.array(X_test)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)  # Ensure the tensor has 3 dimensions (batch_size, seq_len, feature_dim)

# Make predictions
model.eval()
with torch.no_grad():
    predicted_app = model(X_test)
    _, predicted_app_1 = torch.max(predicted_app, 1)

# Confusion matrix (for checking accuracy)
cm = confusion_matrix(test_set, predicted_app_1.numpy())

# Indices of the highest values
idx = (-predicted_app.numpy()).argsort(axis=1)

# Adding randomness to break ties
for i in range(predicted_app.shape[0]):
    np.random.shuffle(idx[i, 1:])

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
