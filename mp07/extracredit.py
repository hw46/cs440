import torch
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE = torch.float32
DEVICE = torch.device("cpu")

def trainmodel():
    # Define the model architecture
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=8*8*15, out_features=1)
    )

    # Initialize the weights
    model[1].weight.data = initialize_weights()
    model[1].bias.data = torch.zeros(1)

    # Load the training dataset
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(20):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

    # Save the trained model
    torch.save(model.state_dict(), 'model_ckpt.pth')

if __name__ == "__main__":
    trainmodel()