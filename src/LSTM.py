import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense, GaussianNoise, Reshape, Lambda
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import pennylane as qml
from pennylane.qnn import KerasLayer

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
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(w)) for w in range(4)]

weight_shapes = {"weights": (2, 4, 3)}

# Create the KerasLayer for the quantum circuit
qlayer = KerasLayer(quantum_circuit, weight_shapes, output_dim=4)

# Function to create the hybrid quantum-classical LSTM model
def create_quantum_lstm_model():
    model = Sequential([
        LSTM(units=256, return_sequences=True, input_shape=(10, 1), name='lstm_1'),
        Dropout(0.5, name='dropout_1'),
        BatchNormalization(name='batch_normalization_1'),
        GaussianNoise(0.1),
        Dense(units=4, activation='linear', name='dense_to_quantum'),  # Reduce the dimension to match the number of qubits
        Lambda(lambda x: tf.reshape(x, (-1, 4))),  # Reshape to (batch_size * timesteps, features)
        qlayer,
        Lambda(lambda x: tf.reshape(x, (-1, 10, 4))),  # Reshape back to (batch_size, timesteps, features)
        LSTM(units=256, return_sequences=True, name='lstm_2'),
        Dropout(0.5, name='dropout_2'),
        BatchNormalization(name='batch_normalization_2'),
        GaussianNoise(0.1),
        LSTM(units=256, return_sequences=True, name='lstm_3'),
        Dropout(0.5, name='dropout_3'),
        BatchNormalization(name='batch_normalization_3'),
        GaussianNoise(0.1),
        LSTM(units=256, name='lstm_4'),
        Dropout(0.5, name='dropout_4'),
        BatchNormalization(name='batch_normalization_4'),
        GaussianNoise(0.1),
        Dense(units=36, activation='softmax', name='dense')
    ])
    return model

# Main script
data = pd.read_csv('Datasets/dataset.csv')
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
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

label_encoder_y = LabelEncoder()
y_train = label_encoder_y.fit_transform(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes=36)

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True)

# Prepare TensorBoard callback
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Compile and train the model
accuracy_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    qlstm_model = create_quantum_lstm_model()
    qlstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    callback_es = EarlyStopping(monitor='val_loss', patience=10)

    qlstm_model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[tensorboard_callback, callback_es])

    scores = qlstm_model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_scores.append(scores[1])

print(f"Cross-validated accuracy scores: {accuracy_scores}")
print(f"Mean accuracy: {np.mean(accuracy_scores)}")
qlstm_model.summary()
# Save the model
qlstm_model.save('quantum_lstm_model.h5')

# Testing the model
X_test = []
inputs = training_set_scaled[len(training_set_scaled) - len(test_set) - 10:].reshape(-1, 1)
inputs = scaler.transform(inputs)

for i in range(10, 397):
    X_test.append(inputs[i-10:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Make predictions
predicted_app = qlstm_model.predict(X_test)
predicted_app_1 = np.argmax(predicted_app, axis=1)

# Confusion matrix (for checking accuracy)
cm = confusion_matrix(test_set, predicted_app_1)

# Indices of the highest values
idx = (-predicted_app).argsort(axis=1)

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

# Run the following command in your terminal to start TensorBoard
# tensorboard --logdir logs/fit
