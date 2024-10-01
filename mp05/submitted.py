import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    block = nn.Sequential(
        nn.Linear(2,3),
        nn.Sigmoid(),
        nn.Linear(3,5)
    )
    return block


def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    return nn.CrossEntropyLoss()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.reshape(-1, 3, 31, 31)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.reshape(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = ConvNet()
    loss_fn = create_loss_function()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.17, weight_decay=0.000095)
    for epoch in range(epochs):
        for images, labels in train_dataloader:
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    ################## Your Code Ends here ##################

    return model

print(torch.__version__)
print(1+2)