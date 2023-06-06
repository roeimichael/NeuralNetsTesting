import pandas as pd
from models.CNN1D import CNN1D
from models.ResNet import resnet18
from models.MLP import MLP
from models.LSTM import LSTM
from main import evaluate_model, prepare_data, train_model
from CSVDataset import CSVDataset
from torch.optim import SGD, Adam, RMSprop
from torch.nn import BCELoss, MSELoss
from lossfunctions.focal import FocalLoss
from lossfunctions.loss_utils import CostSensitiveLoss
from torch.nn import HingeEmbeddingLoss, BCELoss, MSELoss

with open('./data/dates.txt', 'r') as fp:
    dates = [line.strip() for line in fp.readlines()]


def get_top_models():
    # Replace these lists with the top 3 parameter sets for each model type
    top_mlp_params = []
    top_cnn1d_params = []
    top_lstm_params = []
    top_resnet_params = []

    top_models = []
    for params in top_mlp_params:
        top_models.append(('MLP', MLP(**params)))
    for params in top_cnn1d_params:
        top_models.append(('CNN1D', CNN1D(**params)))
    for params in top_lstm_params:
        top_models.append(('LSTM', LSTM(**params)))
    for params in top_resnet_params:
        top_models.append(('resnet18', resnet18(**params)))

    return top_models


def main():
    top_models = get_top_models()
    date_range = range(120, 141)
    results = []
    n_epochs = 50
    # best_loss_function = {
    #     "MLP": BCELoss(),
    #     "CNN1D": BCELoss(),
    #     "LSTM": BCELoss(),
    #     "resnet18": BCELoss()
    # }
    # best_optimizer = {
    #     "MLP": Adam,
    #     "CNN1D": Adam,
    #     "LSTM": Adam,
    #     "resnet18": Adam
    # }
    # best_optimizer_params = {
    #     "MLP": {"lr": 0.001},
    #     "CNN1D": {"lr": 0.001},
    #     "LSTM": {"lr": 0.001},
    #     "resnet18": {"lr": 0.001}
    # }

    for date_index in date_range:

        for model_name, model in top_models:

            path_train = f'./data/dates1.5/{dates[date_index]}.csv'
            path_test = f'./data/dates1.5/{dates[date_index + 1]}.csv'
            next_day_data_path = f'./data/dates1.5/{dates[date_index + 2]}.csv'
            train_dl, test_dl = prepare_data(path_train, path_test)

            for epoch in range(n_epochs):
                train_model(train_dl, model, criterion, optimizer)
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{n_epochs}] done.')

            acc, precision, recall, f1, specificity, mcc, avg_roi = evaluate_model(test_dl, model, next_day_data_path)
            print(
                f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-score: {f1}, Specificity: {specificity}, MCC: {mcc}, Avg ROI: {avg_roi}")

            results.append({'Model': model_name, 'DateIndex': date_index, 'Accuracy': acc, 'Precision': precision,
                            'Recall': recall, 'F1-score': f1, 'Specificity': specificity, 'MCC': mcc,
                            'Avg_ROI': avg_roi})

    results_df = pd.DataFrame(results)
    avg_results = results_df.groupby('Model')['TestAccuracy'].mean().reset_index()
    avg_results.columns = ['Model', 'AverageTestAccuracy']

    print(avg_results)
    avg_results.to_csv('avg_results.csv', index=False)


if __name__ == '__main__':
    main()
