import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score
from torch.optim import SGD, Adam, RMSprop
from torch.nn import HingeEmbeddingLoss, BCELoss, MSELoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from lossfunctions.loss_utils import CostSensitiveLoss
from lossfunctions.focal import FocalLoss
from CSVDataset import CSVDataset
from models.CNN1D import CNN1D
from models.ResNet import resnet18
from models.MLP import MLP
from models.LSTM import LSTM
from models.CNNcmpl import ComplexCNN
from sklearn.metrics import f1_score, matthews_corrcoef
import rtdl
from tqdm import tqdm
import functools


def print_function_name_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


@print_function_name_decorator
def prepare_data(path_train, path_test, train_path, starting_date):
    dataset = CSVDataset(path_train, path_test, train_path, starting_date)
    x_train_tensor = torch.from_numpy(dataset.x_train).float()
    y_train_tensor = torch.from_numpy(dataset.y_train).float()
    x_test_tensor = torch.from_numpy(dataset.x_test).float()
    y_test_tensor = torch.from_numpy(dataset.y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_dl = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

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

    skip_models = []

    train_dl, test_dl = prepare_data(path_train, path_test, train_path, starting_date)
    n_inputs = train_dl.dataset.tensors[0].shape[1]

    models = [
        {"name": "ResNet", "class": resnet18, "params": {}},
        {"name": "LSTM", "class": LSTM,
         "params": {"hidden_dim": [32, 64, 128, 256], "num_layers": [2, 3, 4], "dropout_rate": [0.1, 0.3, 0.5]}},
        {"name": "MLP", "class": MLP,
         "params": {"hidden_dim": [1024, 512, 256, 128], "dropout_rate": [0.1, 0.3, 0.5]}},
        {"name": "CNN1D", "class": CNN1D, "params": {"hidden_dim": [64, 128, 256], "dropout_rate": [0.1, 0.3, 0.5]}},
        {"name": "ComplexCNN", "class": ComplexCNN,
         "params": {"hidden_dim": [32, 64, 128], "dropout_rate": [0.1, 0.3, 0.5]}}
    ]

    y_train = train_dl.dataset.tensors[1]
    n_positive = y_train.sum().item()
    n_negative = len(y_train) - n_positive
    ratio_positive_negative = n_negative / n_positive
    cost_matrix_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, ratio_positive_negative]

    cost_sensitive_loss_functions = [
        {
            "name": f"CostSensitiveLoss_{cost}",
            "function": CostSensitiveLoss(weight=100, cost_matrix=np.array([[cost, 1 - cost], [1 - cost, cost]]),
                                          reduction="mean")}
        for cost in cost_matrix_values
    ]
    optimizers = [
        {"name": "Adam", "class": Adam, "params": {"lr": [0.001, 0.01]}},
        {"name": "RMSprop", "class": RMSprop, "params": {"lr": [0.001, 0.01]}}
    ]
    try:
        for model_info in tqdm(models, desc='Processing models', unit='model'):
            model_name = model_info["name"]
            save_directory = create_save_directory(model_name)

            if model_name in skip_models:
                print(f"Skipping {model_name}...")
                continue

            model_class = model_info["class"]
            model_params = model_info["params"]
            # print(model_params)

            for params in tqdm(ParameterGrid(model_params), desc='Processing params', unit='param'):

                for loss_function_info in tqdm(cost_sensitive_loss_functions, desc='loss_functions',
                                               unit='loss_function'):

                    cost_matrix_value = float(loss_function_info["name"].split('_')[1])  # Extract the cost_matrix_value
                    loss_function = loss_function_info["function"]

                    for optimizer_info in tqdm(optimizers, desc='optimizers', unit='optimizer'):

                        optimizer_name = optimizer_info["name"]
                        optimizer_class = optimizer_info["class"]
                        optimizer_params = optimizer_info["params"]

                        for optimizer_p in ParameterGrid(optimizer_params):

                            if model_name == "MLP":
                                model = model_class(n_inputs, params['hidden_dim'], params["dropout_rate"]).to(device)
                            elif model_name == "CNN1D":
                                model = model_class(n_inputs, params['hidden_dim'], params["dropout_rate"]).to(device)
                            elif model_name == "ResNet":
                                model = model_class().to(device)
                            elif model_name == "LSTM":
                                model = model_class(n_inputs, params["hidden_dim"], params["num_layers"],
                                                    params["dropout_rate"]).to(device)
                            elif model_name == "ComplexCNN":
                                model = model_class(n_inputs, params['hidden_dim'], params["dropout_rate"]).to(device)
                            criterion = loss_function
                            optimizer = optimizer_class(model.parameters(), **optimizer_p)
                            for epoch in range(n_epochs):
                                if epoch % 10 == 0:
                                    print(f"Epoch: {epoch}")
                                train_model(train_dl, model, criterion, optimizer, device)

                            save_model(model, model_name, params, save_directory)
                            loss, acc, precision, recall, f1, mcc, avg_roi, true_positive_positions, opened_positions, positive_positions = evaluate_model(
                                test_dl, model, criterion, next_day_data_path, device)
                            # print( f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-score: {f1}
                            # MCC: {mcc}, Avg ROI: {avg_roi}")
                            result = {
                                "Model": model_name,
                                "Model_Params": params,
                                "Cost_Matrix_Value": cost_matrix_value,
                                "Optimizer": optimizer_name,
                                "Optimizer_Params": optimizer_p,
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
                            results.to_csv("intermediate_results.csv", index=False)

    except KeyboardInterrupt:
        print("\nStopping the code execution...")
    results.to_csv("model_performance_results.csv", index=False)


if __name__ == "__main__":
    main()
