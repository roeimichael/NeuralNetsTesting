import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from CSVDataset import CSVDataset
from torchvision.models import resnet18
import torch.optim as optim


def prepare_data(path_train, path_test, train_path, starting_date):
    dataset = CSVDataset(path_train, path_test, train_path, starting_date)
    x_train_tensor = torch.from_numpy(dataset.x_train).float().view(-1, 1, 11, 45)
    y_train_tensor = torch.from_numpy(dataset.y_train).float()
    x_test_tensor = torch.from_numpy(dataset.x_test).float().view(-1, 1, 11, 45)
    y_test_tensor = torch.from_numpy(dataset.y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_dl, test_dl


def train_model(train_dl, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for i, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, 1), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_dl)


def evaluate_model(test_dl, model, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 1), targets)
            running_loss += loss.item()
    return running_loss / len(test_dl)


def main(n_epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    with open('./data/dates.txt', 'r') as fp:
        dates = [line.strip() for line in fp.readlines()]
    starting_date = 466
    path_train = f'./data/dates1.5/{dates[starting_date]}.csv'
    path_test = f'./data/dates1.5/{dates[starting_date + 1]}.csv'
    train_path = './data/Targets1.5.csv'
    next_day_data_path = f'./data/dates1.5/{dates[starting_date + 2]}.csv'
    results = pd.DataFrame(columns=["Model", "Model_Params", "Cost_Matrix_Value", "Optimizer", "Optimizer_Params",
                                    "Accuracy", "Precision", "Recall", "F1-score", "MCC", "Avg_ROI", "TP_positions",
                                    "total_positions", "positive_positions"])

    train_dl, test_dl = prepare_data(path_train, path_test, train_path, starting_date)

    # Create the model
    model = resnet18()  # Create the model
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust first layer
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(n_epochs):
        train_loss = train_model(train_dl, model, criterion, optimizer, device)
        test_loss = evaluate_model(test_dl, model, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "trained_resnet.pt")


if __name__ == "__main__":
    main()
