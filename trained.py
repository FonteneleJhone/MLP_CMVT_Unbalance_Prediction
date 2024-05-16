import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the MLP model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc_input(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.fc_output(x)
        return torch.sigmoid(x)

# Read data from CSV file
data_to_predict = pd.read_csv('saudavel.csv', delimiter=';')

# Extract features
X_to_predict = data_to_predict[['X_Axis', 'Y_Axis', 'Z_Axis', 'Sum_Vectors']].values

# Standardize features
scaler = StandardScaler()
X_to_predict = scaler.fit_transform(X_to_predict)

# Load the trained model
input_size = 4
hidden_size = 10
output_size = 1
num_layers = 5  # Update with the correct number of hidden layers used in training
model = MLP(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    inputs = torch.tensor(X_to_predict, dtype=torch.float32)
    outputs = model(inputs)
    predictions = (outputs.squeeze().numpy() >= 0.5).astype(int)

# Map predictions to balanced or unbalanced behavior
behavior_mapping = {0: 'Balanced', 1: 'Unbalanced'}
predicted_behavior = [behavior_mapping[prediction] for prediction in predictions]

# Print predicted behavior
for i, behavior in enumerate(predicted_behavior):
    print(f'Data point {i+1}: {behavior}')
