import os
import numpy as np

# Load the pre-trained Quantum LSTM model
def getTrainedModel(modelFilePath='./../Models/trainedModel.json', weightFilePath='./../Models/weights.h5'):
    current_file_path = os.path.dirname(__file__)
    with open(os.path.join(current_file_path, modelFilePath), 'r') as json_file:
        model_data = json.load(json_file)
    weights = np.array(model_data['weights'])
    return weights
