import os
import json
from datetime import datetime
import numpy as np

# Saving the weights and biases of the trained Quantum LSTM model.
def saveRNNModel(QuantumLSTMModel):
    current_file_path = os.path.dirname(__file__)
    currentTime = datetime.now()
    formattedTimeinString = currentTime.strftime('%d%m%Y%H%M%S')
    os.makedirs(os.path.join(current_file_path, '../Models/' + formattedTimeinString))
    new_file_path = os.path.join(current_file_path, '../Models/' + formattedTimeinString)
    model_data = {'weights': QuantumLSTMModel.tolist()}
    with open(os.path.join(new_file_path, 'trainedModel.json'), "w") as json_file:
        json.dump(model_data, json_file)
