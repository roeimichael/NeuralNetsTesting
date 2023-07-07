import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef


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


class ModelConfig:
    def __init__(self, model_info, params, loss_function_info, optimizer_info, optimizer_p, cost_matrix_value,
                 train_dl, test_dl, n_epochs, next_day_data_path, device, save_directory, model_class, optimizer_class):
        self.model_info = model_info
        self.params = params
        self.loss_function_info = loss_function_info
        self.optimizer_info = optimizer_info
        self.optimizer_p = optimizer_p
        self.cost_matrix_value = cost_matrix_value
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.n_epochs = n_epochs
        self.next_day_data_path = next_day_data_path
        self.device = device
        self.n_inputs = train_dl.dataset.tensors[0].shape[1]
        self.save_directory = save_directory
        self.model_class = model_class
        self.optimizer_class = optimizer_class

    def run_model(self):
        model_name = self.model_info["name"]
        model_class = self.model_info["class"]
        loss_function = self.loss_function_info["function"]
        optimizer_class = self.optimizer_info["class"]
        if model_name == "MLP":
            model = model_class(self.n_inputs, self.params['hidden_dim'], self.params["dropout_rate"]).to(self.device)
        elif model_name == "CNN1D":
            model = model_class(self.n_inputs, self.params['hidden_dim'], self.params["dropout_rate"]).to(self.device)
        elif model_name == "ResNet":
            model = model_class(self.params['hidden_dim'], [2, 2, 2, 2], 1).to(self.device)
        elif model_name == "LSTM":
            model = model_class(self.n_inputs, self.params["hidden_dim"], self.params["num_layers"],
                                self.params["dropout_rate"]).to(self.device)
        elif model_name == "ComplexCNN":
            model = model_class(self.n_inputs, self.params['hidden_dim'], self.params["dropout_rate"]).to(self.device)
        criterion = loss_function
        optimizer = optimizer_class(model.parameters(), **self.optimizer_p)

        for epoch in range(self.n_epochs):
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}")
            train_model(self.train_dl, model, criterion, optimizer, self.device)

        save_model(model, model_name, self.params, self.save_directory)
        loss, acc, precision, recall, f1, mcc, avg_roi, true_positive_positions, opened_positions, positive_positions = evaluate_model(
            self.test_dl, model, criterion, self.next_day_data_path, self.device)

        return {
            "Model": model_name,
            "Model_Params": self.params,
            "Cost_Matrix_Value": self.cost_matrix_value,
            "Optimizer": self.optimizer_info["name"],
            "Optimizer_Params": self.optimizer_p,
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
