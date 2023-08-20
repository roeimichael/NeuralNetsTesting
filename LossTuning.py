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
import torchvision.models as models


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

def generate_model_name(base_name, params):
    """
    Generate a unique model name based on the hyperparameters.
    """
    params_str = "_".join(f"{k}={v}" for k, v in params.items())
    return f"{base_name}_{params_str}"

def adjust_cost(positive_positions, target=25, delta=2, step=0.01, lower_bound=0.0, upper_bound=1.0):
    """Adjust the cost based on the number of positive predictions."""

    # Adjust cost based on target and positive positions
    if positive_positions > target + delta:
        adjustment = step
    elif positive_positions < target - delta:
        adjustment = -step
    else:
        return None

    # Ensure cost stays within bounds and has only two decimal points
    return round(max(lower_bound, min(upper_bound, adjustment)), 2)


@print_function_name_decorator
def main(n_epochs=200):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs Available: {num_gpus}")
        for gpu_id in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(gpu_id)
            print(f"GPU {gpu_id}: {gpu_name}")

        device = torch.device("cuda:3")
    else:
        print("No GPUs available, using the CPU instead.")
        device = torch.device("cpu")
    print(f"Current device in use: {device}")

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

    results = pd.DataFrame(
        columns=["Hidden_Size", "Num_Blocks", "Cost_Matrix_Value", "Accuracy", "Precision", "Recall", "F1-score", "MCC",
                 "Avg_ROI", "TP_positions", "total_positions", "positive_positions"])

    hyperparams = [
        {"hidden_size": 128, "num_blocks": [2, 2, 2, 2]},
        {"hidden_size": 256, "num_blocks": [2, 2, 2, 2]},
        {"hidden_size": 128, "num_blocks": [3, 3, 3, 3]},
        {"hidden_size": 64, "num_blocks": [2, 2, 2, 2]},
        {"hidden_size": 512, "num_blocks": [2, 2, 2, 2]},
        {"hidden_size": 1024, "num_blocks": [2, 2, 2, 2]},
        {"hidden_size": 128, "num_blocks": [1, 2, 3, 4]},
        {"hidden_size": 128, "num_blocks": [4, 3, 2, 1]},
        {"hidden_size": 128, "num_blocks": [2, 3, 3, 2]},
    ]

    for params in hyperparams:
        hidden_size = params["hidden_size"]
        num_blocks = params["num_blocks"]

        print(f"Running for hyperparameters - Hidden Size: {hidden_size}, Num Blocks: {num_blocks}")
        model_name = generate_model_name("ResNet", params)
        save_directory = create_save_directory(model_name)

        try:
            model = ResNet(hidden_size, num_blocks, 1).to(device)
            optimizer = Adam(model.parameters(), lr=0.00001)

            # Loop until the model reaches the desired number of positive predictions
            while True:
                # Defining the loss function with the current cost value
                loss_function = CostSensitiveLoss(weight=100,
                                                  cost_matrix=np.array([[1 - cost, cost], [cost, 1 - cost]]),
                                                  reduction="mean")

                # Inside your training loop:
                for epoch in range(n_epochs):
                    # Training the model
                    train_model(train_dl, model, loss_function, optimizer, device)

                    # Printing the epoch number every 50 epochs
                    if (epoch + 1) % 50 == 0:
                        print(f"Epoch {epoch + 1}/{n_epochs} completed")

                # Evaluating the model
                _, _, _, _, _, _, _, true_positive_positions, opened_positions, positive_positions = evaluate_model(
                    test_dl, model, loss_function, next_day_data_path, device)

                cost_adjustment = adjust_cost(positive_positions)
                if cost_adjustment is None:
                    save_model(model, model_name, params, save_directory)

                    break
                cost += cost_adjustment
                print(f"Number of positive predictions: {positive_positions}")
                print(f"Updated cost value: {cost}")

            # Adding final results
            loss, acc, precision, recall, f1, mcc, avg_roi, true_positive_positions, opened_positions, positive_positions = evaluate_model(
                test_dl, model, loss_function, next_day_data_path, device)
            result = {"Hidden_Size": hidden_size, "Num_Blocks": '_'.join(map(str, num_blocks)),
                      "Cost_Matrix_Value": cost, "Accuracy": acc, "Precision": precision, "Recall": recall,
                      "F1-score": f1, "MCC": mcc, "Avg_ROI": avg_roi, "TP_positions": true_positive_positions,
                      "total_positions": opened_positions, "positive_positions": positive_positions}
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

        except KeyboardInterrupt:
            print("\nStopping the code execution...")

    results.to_csv("finalresults.csv", index=False)


if __name__ == "__main__":
    main()
