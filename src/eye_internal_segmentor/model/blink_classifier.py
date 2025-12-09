import torch

class blinker(torch.nn.Module):
    def __init__(self, num_inputs, dropout=False):
        super().__init__()
        self.dropout_bool = dropout
        self.flatten = torch.nn.Flatten()

        if dropout:
            print('BLINK Classifier dropout is True')
            self.dropout = torch.nn.Dropout(p=0.4)
        else:
            self.dropout = None

        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_inputs, out_features=256),
            torch.nn.ReLU()
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear1(self.flatten(x))

        if self.dropout_bool:
            x = self.dropout(x)

        x = self.linear2(x)
        return x