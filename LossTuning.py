import pandas as pd
import numpy as np
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from lossfunctions.loss_utils import CostSensitiveLoss
from CSVDataset import CSVDataset
from models.ResNet import ResNet
import functools
import multiprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score


def print_function_name_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def prepare_data(path_train, path_test, train_path, starting_date):
    dataset = CSVDataset(path_train, path_test, train_path, starting_date)
    x_train_tensor = torch.from_numpy(dataset.x_train).float()
    y_train_tensor = torch.from_numpy(dataset.y_train).float()
    x_test_tensor = torch.from_numpy(dataset.x_test).float()
    y_test_tensor = torch.from_numpy(dataset.y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    num_workers = multiprocessing.cpu_count()
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_dl, test_dl


def train_model(train_dl, model, criterion, optimizer, device):
    for i, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        yhat = model(inputs)
        loss = criterion(yhat, targets)
        loss.backward()
        optimizer.step()


def evaluate_model(test_dl, model, criterion, next_day_data_path, device):
    next_day_data = pd.read_csv(next_day_data_path)['Close Change']
    predictions, actuals = [], []
    total_loss = 0  # Initialize total loss

    for inputs, targets in test_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs)
        loss = criterion(yhat, targets)  # Calculate the loss for this batch
        total_loss += loss.item()  # Accumulate the losses

        yhat = yhat.detach().cpu().numpy()
        predictions.extend(yhat)
        actuals.extend(targets.cpu().numpy().reshape(-1, 1))

    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    opened_positions = (predictions.round() == 1).sum()
    true_positive_positions = np.sum((predictions.round().flatten() == 1) & (actuals.flatten() == 1))
    positive_positions = (next_day_data[predictions.round().flatten() == 1] > 0).sum()
    total_roi = next_day_data[predictions.round().flatten() == 1].sum()
    average_roi = total_roi / opened_positions if opened_positions > 0 else 0

    avg_loss = total_loss / len(test_dl)  # Calculate the average loss

    return (
        avg_loss,
        accuracy_score(actuals, predictions.round()),
        precision_score(actuals, predictions.round(), zero_division=0),
        recall_score(actuals, predictions.round()),
        f1_score(actuals, predictions.round(), zero_division=0),
        matthews_corrcoef(actuals, predictions.round(), sample_weight=None),
        average_roi,
        true_positive_positions,
        opened_positions,
        positive_positions
    )


def create_save_directory(model_name):
    save_directory = f"./saved_models/{model_name}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    return save_directory


def save_model(model, model_name, params, save_directory):
    model_directory = os.path.join(save_directory, model_name)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    params_str = "_".join(f"{k}={v}" for k, v in params.items())
    filename = f"{model_directory}/final_model_{model_name}_{params_str}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params
    }, filename)


def load_model(model, model_name, epoch, save_directory):
    checkpoint = torch.load(f"{save_directory}/{model_name}_epoch_{epoch}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


@print_function_name_decorator
def main(n_epochs=50):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        print("GPU Name:", gpu_properties.name)
        print("GPU Capability:", gpu_properties.major, gpu_properties.minor)
        print("Total Memory (GB):", gpu_properties.total_memory / 1e9)
    else:
        print("CUDA is not available on this system.")
    with open('./data/dates.txt', 'r') as fp:
        dates = [line.strip() for line in fp.readlines()]
    starting_date = 466
    path_train = f'./data/dates1.5/{dates[starting_date]}.csv'
    path_test = f'./data/dates1.5/{dates[starting_date + 1]}.csv'
    train_path = './data/Targets1.5.csv'
    next_day_data_path = f'./data/dates1.5/{dates[starting_date + 2]}.csv'
    train_dl, test_dl = prepare_data(path_train, path_test, train_path, starting_date)
    initial_cost_value = 0.5
    cost = initial_cost_value

    results = pd.DataFrame(columns=["Cost_Matrix_Value", "Accuracy", "Precision", "Recall", "F1-score", "MCC",
                                    "Avg_ROI", "TP_positions", "total_positions", "positive_positions"])

    try:
        model = ResNet(64, [2, 2, 2, 2], 1).to(device)
        optimizer = Adam(model.parameters(), lr=0.0001)

        # Loop until the model reaches the desired number of positive predictions
        while True:
            # Defining the loss function with the current cost value
            loss_function = CostSensitiveLoss(weight=100,
                                              cost_matrix=np.array([[cost, 1 - cost], [1 - cost, cost]]),
                                              reduction="mean")

            # Training the model
            for epoch in range(n_epochs):
                if epoch % 10 == 0:
                    print(f"Epoch: {epoch}")
                train_model(train_dl, model, loss_function, optimizer, device)

            # Evaluating the model
            _, _, _, _, _, _, _, true_positive_positions, opened_positions, positive_positions = evaluate_model(
                test_dl, model, loss_function, next_day_data_path, device)

            # Adjusting the cost value based on the number of positive predictions
            target_positive_predictions = 20
            if positive_positions > target_positive_predictions:
                # If positive predictions are above 20, increase the cost
                cost += 0.01
            elif positive_positions < target_positive_predictions:
                # If positive predictions are below 20, decrease the cost
                cost -= 0.01
            else:
                # If positive predictions are exactly 20, stop the loop
                break

            # Ensure the cost is within valid bounds
            cost = max(0.0, min(1.0, cost))
            print(f"Number of positive predictions: {positive_positions}")
            print(f"Updated cost value: {cost}")

        # Adding final results
        loss, acc, precision, recall, f1, mcc, avg_roi, true_positive_positions, opened_positions, positive_positions = evaluate_model(
            test_dl, model, loss_function, next_day_data_path, device)
        result = {
            "Cost_Matrix_Value": cost,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "MCC": mcc,
            "Avg_ROI": avg_roi,
            "TP_positions": true_positive_positions,
            "total_positions": opened_positions,
            "positive_positions": positive_positions
        }
        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
        results.to_csv("results.csv", index=False)

    except KeyboardInterrupt:
        print("\nStopping the code execution...")
    results.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
