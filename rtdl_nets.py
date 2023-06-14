import pandas as pd
import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from CSVDataset_rtdl import CSVDataset_rtdl
from torch.utils.data import DataLoader, TensorDataset
from models.rtdl_wrapper import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')

zero.improve_reproducibility(seed=123456)

def prepare_data(path_train, path_test, train_path, starting_date):
    dataset = CSVDataset_rtdl(path_train, path_test, train_path, starting_date)
    data = dataset.get_data()
    preprocess_num = sklearn.preprocessing.StandardScaler().fit(data['train'][0])
    preprocess_cat = sklearn.preprocessing.OrdinalEncoder().fit(data['train'][1])
    data = {
        k: (torch.tensor(preprocess_num.transform(v[0]), device=device),
            torch.tensor(preprocess_cat.transform(v[1]), device=device).long(),
            torch.tensor(v[2], device=device))
        for k, v in data.items()
    }
    return data, dataset.cardinalities_train, dataset.cardinalities_test


with open('./data/dates.txt', 'r') as fp:
    dates = [line.strip() for line in fp.readlines()]

starting_date = 466
path_train = f'./data/dates1.5/{dates[starting_date]}.csv'
path_test = f'./data/dates1.5/{dates[starting_date + 1]}.csv'
train_path = './data/Targets1.5.csv'
next_day_data_path = f'./data/dates1.5/{dates[starting_date + 2]}.csv'

data, cardinalities_train, cardinalities_test = prepare_data(path_train, path_test, train_path, starting_date)
train_dataset = TensorDataset(*data['train'])
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(*data['test'])
test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False)

n_inputs = data['train'][0].shape[1]
d_token, d_out = 2, 1

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    prediction = []
    for num_features, cat_features, targets in dataloader:
        prediction.append(model(num_features, cat_features))

    prediction = torch.cat(prediction).cpu().numpy()
    target = dataloader.dataset.tensors[2].cpu().numpy().flatten()
    prediction = scipy.special.expit(prediction)
    prediction = np.round(prediction)

    accuracy = accuracy_score(target, prediction)
    precision = precision_score(target, prediction)
    recall = recall_score(target, prediction)
    f1 = f1_score(target, prediction)
    confusion = confusion_matrix(target, prediction)

    return accuracy, precision, recall, f1, confusion

def train_and_evaluate(layers, layer_size, dropout, lr, weight_decay):
    model = Model(
        n_inputs,
        rtdl.CategoricalFeatureTokenizer(cardinalities_train, d_token, True, 'uniform'),
        {"d_layers": [layer_size] * layers, "dropout": dropout, "d_out": d_out})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = F.binary_cross_entropy_with_logits

    batch_size = 256
    train_loader = zero.data.IndexLoader(len(train_dl), batch_size, device=device)
    progress = zero.ProgressTracker(patience=100)
    n_epochs = 100

    print(f'Training Model with {layers} layers of size {layer_size}, dropout {dropout}, learning rate {lr}, weight decay {weight_decay}')
    for epoch in range(1, n_epochs + 1):
        for iteration, (num_features, cat_features, targets) in enumerate(train_dl):
            model.train()
            optimizer.zero_grad()
            output = model(num_features, cat_features)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

        final_metrics = evaluate(model, test_dl)
        progress.update(final_metrics[0])
        if progress.fail:
            break

    final_metrics = evaluate(model, test_dl)
    return final_metrics

layers_to_test = [3, 5, 7, 10]
layer_sizes_to_test = [64, 128, 256]
dropout_to_test = [0.1, 0.3, 0.5]
lr_to_test = [0.001, 0.01]
weight_decay_to_test = [0.0, 0.1, 0.2]

results_df = pd.DataFrame()

for layers in layers_to_test:
    for layer_size in layer_sizes_to_test:
        for dropout in dropout_to_test:
            for lr in lr_to_test:
                for weight_decay in weight_decay_to_test:
                    final_metrics = train_and_evaluate(layers, layer_size, dropout, lr, weight_decay)

                    new_results = pd.DataFrame({
                        'Model': ['ResNet'],
                        'Layers': [layers],
                        'Layer Size': [layer_size],
                        'Dropout': [dropout],
                        'Learning Rate': [lr],
                        'Weight Decay': [weight_decay],
                        'Accuracy': [final_metrics[0]],
                        'Precision': [final_metrics[1]],
                        'Recall': [final_metrics[2]],
                        'F1 Score': [final_metrics[3]],
                        'Confusion Matrix': [final_metrics[4]]
                    })

                    results_df = pd.concat([results_df, new_results], ignore_index=True)
                    results_df.to_csv('model_results.csv', index=False)

# Determine the number of inputs based on the size of the training data
# n_inputs = data['train'][0].shape[1]
# d_token, d_out = 2, 1
# # creates the model
# model = Model(
#     n_inputs,
#     rtdl.CategoricalFeatureTokenizer(cardinalities_train, d_token, True, 'uniform'),
#     {
#         "d_layers": [256]*10,  # Ten layers each of size 256
#         "dropout": 0.1,
#         "d_out": d_out
#     })

# setting the parameters for the calculation
# lr = 0.001
# weight_decay = 0.0
# model.to(device)
# optimizer = (
#     model.make_default_optimizer()
#     if isinstance(model, rtdl.FTTransformer)
#     else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# )
# loss_fn = (
#     F.binary_cross_entropy_with_logits
# )


# Define an evaluation function to calculate accuracy on a given dataset part (e.g., 'test' or 'val')


# new_results = pd.DataFrame({
#     'Model': ['ResNet'],
#     'Layers': [10],
#     'Layer Size': [256],
#     'Dropout': [0.1],
#     'Learning Rate': [lr],
#     'Weight Decay': [weight_decay],
#     'Accuracy': [final_metrics[0]],
#     'Precision': [final_metrics[1]],
#     'Recall': [final_metrics[2]],
#     'F1 Score': [final_metrics[3]],
#     'Confusion Matrix': [final_metrics[4]]
# })
#
# results_df = pd.concat([results_df, new_results], ignore_index=True)
