from pandas import read_csv

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CSVDataset(Dataset):
    def __init__(self, path_train, path_test, targets_path, starting_date):
        df_train = self.read_and_preprocess(path_train)
        df_test = self.read_and_preprocess(path_test)
        self.targets_df = read_csv(targets_path, index_col=0)

        self.x_train, self.y_train = self.split_features_labels(df_train, starting_date + 1)
        self.x_test, self.y_test = self.split_features_labels(df_test, starting_date + 2)

        # Normalize the training and test data using StandardScaler
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    def read_and_preprocess(self, path):
        df = read_csv(path, header=None)
        df = df.iloc[1:, 1:]
        return df

    def split_features_labels(self, df, target_date):
        x = df.values[:, 1:].astype('float32')
        y = self.targets_df.values[:, target_date]
        y = LabelEncoder().fit_transform(y)
        y = y.astype('float32').reshape((len(y), 1))
        return x, y

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return [self.x_train[idx], self.y_train[idx]]
