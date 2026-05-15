import torch
import torch.nn as nn 
import torch.nn.functional as F 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(784, 100)
        self.l2 = nn.Linear(100,50)
        self.l3 = nn.Linear(50,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(x)

        return x
if __name__ == "__main__":
    NeuralNet = NeuralNetwork()

