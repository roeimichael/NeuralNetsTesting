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


# creates dates list
with open('./data/dates.txt', 'r') as fp:
    dates = [line.strip() for line in fp.readlines()]

# first date to check
starting_date = 466

# fixing paths to correct folders
path_train = f'./data/dates1.5/{dates[starting_date]}.csv'
path_test = f'./data/dates1.5/{dates[starting_date + 1]}.csv'
train_path = './data/Targets1.5.csv'
next_day_data_path = f'./data/dates1.5/{dates[starting_date + 2]}.csv'

# Prepare the data using the function defined above
data, cardinalities_train, cardinalities_test = prepare_data(path_train, path_test, train_path, starting_date)
train_dataset = TensorDataset(*data['train'])
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(*data['test'])
test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Determine the number of inputs based on the size of the training data
n_inputs = data['train'][0].shape[1]
d_token, d_out = 2, 1
# creates the model
model = Model(
    n_inputs,
    rtdl.CategoricalFeatureTokenizer(cardinalities_train, d_token, True, 'uniform'),
    {
        "d_layers": [128, 256, 128],
        "dropout": 0.1,
        "d_out": d_out
    })

# setting the parameters for the calculation
lr = 0.001
weight_decay = 0.0
model.to(device)
optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)
loss_fn = (
    F.binary_cross_entropy_with_logits
)


# Define an evaluation function to calculate accuracy on a given dataset part (e.g., 'test' or 'val')
@torch.no_grad()
def evaluate(dataloader):
    model.eval()
    prediction = []
    for num_features, cat_features, targets in dataloader:
        prediction.append(model(num_features, cat_features))

    prediction = torch.cat(prediction).cpu().numpy()
    target = dataloader.dataset.tensors[2].cpu().numpy().flatten()
    prediction = scipy.special.expit(prediction)
    prediction = np.round(prediction)  # Round off to get the binary predictions
    score = sklearn.metrics.accuracy_score(target, prediction)
    return score


batch_size = 256
train_loader = zero.data.IndexLoader(len(train_dl), batch_size, device=device)
progress = zero.ProgressTracker(patience=100)
print(f'Test score before training: {evaluate(test_dl):.4f}')
n_epochs = 100
report_frequency = 5

for epoch in range(1, n_epochs + 1):
    for iteration, (num_features, cat_features, targets) in enumerate(train_dl):
        model.train()
        optimizer.zero_grad()
        output = model(num_features, cat_features)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

    test_score = evaluate(test_dl)
    print(f'Epoch {epoch:03d} | Test score: {test_score:.4f}', end='')
    progress.update(test_score)
    if progress.success:
        print(' <<< BEST EPOCH', end='')
    print()
    if progress.fail:
        break
