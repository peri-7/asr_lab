import torch
import torch.nn as nn


class TorchDNN(nn.Module):
    """Create a DNN to extract posteriors that can be used for HMM decoding
    Parameters:
        input_dim (int): Input features dimension
        output_dim (int): Number of classes
        num_layers (int): Number of hidden layers
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer
        hidden_dim (int): Number of neurons in each hidden layer
        dropout_p (float): Dropout probability for regularization
    """
    def __init__(
        self, input_dim, output_dim, num_layers=2, batch_norm=True, hidden_dim=256, dropout_p=0.2
    ):
        super(TorchDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim)) # linear
            if batch_norm == True: layers.append(nn.BatchNorm1d(hidden_dim)) # batch norm.
            layers.append(nn.ReLU()) # relus
            layers.append(nn.Dropout(dropout_p)) # dropout
            current_dim = hidden_dim

        # last layer
        layers.append(nn.Linear(current_dim, output_dim))
        # save layers
        self.layers = nn.Sequential(*layers)



    def forward(self, x):
        return self.layers(x)
