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
from sklearn.metrics import f1_score, matthews_corrcoef
import rtdl

# Put prepare_data function here
def prepare_data(path_train, path_test, train_path, starting_date):
    dataset = CSVDataset(path_train, path_test, train_path, starting_date)
    x_train_tensor = torch.from_numpy(dataset.x_train).float()
    y_train_tensor = torch.from_numpy(dataset.y_train).float()
    x_test_tensor = torch.from_numpy(dataset.x_test).float()
    y_test_tensor = torch.from_numpy(dataset.y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    return train_dl, test_dl


def train_model(train_dl, model, criterion, optimizer):
    for i, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        yhat = model(inputs)
        loss = criterion(yhat, targets)
        loss.backward()
        optimizer.step()


def evaluate_model(test_dl, model, next_day_data_path):
    next_day_data = pd.read_csv(next_day_data_path)['Close Change']
    predictions, actuals = [], []

    for inputs, targets in test_dl:
        yhat = model(inputs).detach().numpy()
        predictions.extend(yhat)
        actuals.extend(targets.numpy().reshape(-1, 1))
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    opened_positions = (predictions.round() == 1).sum()
    true_positive_positions = np.sum((predictions.round().flatten() == 1) & (actuals.flatten() == 1))
    positive_positions = (next_day_data[predictions.round().flatten() == 1] > 0).sum()
    total_roi = next_day_data[predictions.round().flatten() == 1].sum()
    average_roi = total_roi / opened_positions if opened_positions > 0 else 0

    print(f"Confusion matrix:\n{confusion_matrix(actuals, predictions.round())}")
    print(f"Total daily ROI: {average_roi:.5f}")
    print(f"Total positive positions: {positive_positions}")

    return (
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


def save_model(model, model_name, params, epoch, save_directory):
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'epoch': epoch
    }, f"{save_directory}/{model_name}_epoch_{epoch}.pth")


def load_model(model, model_name, epoch, save_directory):
    checkpoint = torch.load(f"{save_directory}/{model_name}_epoch_{epoch}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main(n_epochs=100):
    with open('./data/dates.txt', 'r') as fp:
        dates = [line.strip() for line in fp.readlines()]
    starting_date = 466
    path_train = f'./data/dates1.5/{dates[starting_date]}.csv'
    path_test = f'./data/dates1.5/{dates[starting_date + 1]}.csv'
    train_path = './data/Targets1.5.csv'
    next_day_data_path = f'./data/dates1.5/{dates[starting_date + 2]}.csv'
    results = pd.DataFrame(columns=["Model", "Model_Params", "Loss_Function", "Optimizer", "Optimizer_Params",
                                    "Accuracy", "Precision", "Recall", "F1-score", "MCC", "Avg_ROI", "TP_positions",
                                    "total_positions", "positive_positions"])
    skip_models = []

    train_dl, test_dl = prepare_data(path_train, path_test, train_path, starting_date)
    n_inputs = train_dl.dataset.tensors[0].shape[1]

    models = [
        {"name": "MLP", "class": MLP, "params": {"dropout_rate": [0.1, 0.3, 0.5]}},
        {"name": "ResNet", "class": resnet18, "params": {}},
        {"name": "LSTM", "class": LSTM,
         "params": {"hidden_dim": [64], "num_layers": [1, 2], "dropout_rate": [0.1, 0.3, 0.5]}},
        {"name": "CNN1D", "class": CNN1D, "params": {"dropout_rate": [0.1, 0.3, 0.5]}}
    ]
    loss_functions = [
        {"name": "CostSensitiveLoss",
         "function": CostSensitiveLoss(weight=1000, cost_matrix=np.array([[0.3, 0.7], [0.7, 0.3]]), reduction="mean")},
        {"name": "BCELoss", "function": BCELoss()},
        {"name": "MSELoss", "function": MSELoss()},
        {"name": "FocalLoss", "function": FocalLoss(alpha=0.25, gamma=2.0)},
        {"name": "HingeEmbeddingLoss", "function": HingeEmbeddingLoss()}
    ]
    optimizers = [
        {"name": "SGD", "class": SGD, "params": {"lr": [0.01], "momentum": [0.9]}},
        {"name": "Adam", "class": Adam, "params": {"lr": [0.001, 0.01]}},
        {"name": "RMSprop", "class": RMSprop, "params": {"lr": [0.001, 0.01]}}
    ]
    try:
        for model_info in models:
            model_name = model_info["name"]
            if model_name in skip_models:
                print(f"Skipping {model_name}...")
                continue

            model_class = model_info["class"]
            model_params = model_info["params"]
            print(model_params)

            for params in ParameterGrid(model_params):
                print(f"Training {model_name} with params: {params}")

                save_directory = create_save_directory(model_name)

                for loss_function_info in loss_functions:
                    loss_function_name = loss_function_info["name"]
                    loss_function = loss_function_info["function"]
                    for optimizer_info in optimizers:
                        optimizer_name = optimizer_info["name"]
                        optimizer_class = optimizer_info["class"]
                        optimizer_params = optimizer_info["params"]

                        for optimizer_p in ParameterGrid(optimizer_params):
                            print(f"Using optimizer: {optimizer_name} with params: {optimizer_p}")

                            print(f"Using loss function: {loss_function_name}")

                            if model_name == "MLP":
                                model = model_class(n_inputs, params["dropout_rate"])
                            elif model_name == "CNN1D":
                                model = model_class(n_inputs, params["dropout_rate"])
                            elif model_name == "ResNet":
                                model = model_class()
                            elif model_name == "LSTM":
                                model = model_class(n_inputs, params["hidden_dim"], params["num_layers"],
                                                    params["dropout_rate"])

                            criterion = loss_function
                            optimizer = optimizer_class(model.parameters(), **optimizer_p)

                            for epoch in range(n_epochs):
                                train_model(train_dl, model, criterion, optimizer)
                                if (epoch + 1) % 20 == 0:
                                    print(f'Epoch [{epoch}] done.')

                                if (epoch + 1) % 10 == 0:
                                    save_model(model, model_name, params, epoch, save_directory)

                            acc, precision, recall, f1, mcc, avg_roi, true_positive_positions, opened_positions, positive_positions = evaluate_model(
                                test_dl, model, next_day_data_path)
                            print(
                                f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-score: {f1}  MCC: {mcc}, Avg ROI: {avg_roi}")
                            result = {
                                "Model": model_name,
                                "Model_Params": params,
                                "Loss_Function": loss_function_name,
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
    except KeyboardInterrupt:
        print("\nStopping the code execution...")
        results.to_csv("intermediate_results.csv", index=False)
        print("Results saved to intermediate_results.csv")

    results.to_csv("model_performance_results.csv", index=False)


if __name__ == "__main__":
    main()
