import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, in_feature, out_feature) -> None:
        super(MLP, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        layers = []
        layers.append(nn.Linear(in_feature, out_feature))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(out_feature, out_feature))
        # layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x): #(batch_size, input_length, num)
        for layer in self.layers:
            x = layer(x)
        return(x)