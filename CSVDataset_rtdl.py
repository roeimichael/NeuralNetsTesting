from pandas import read_csv
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler


class CSVDataset_rtdl(Dataset):
    def __init__(self, path_train, path_test, targets_path, starting_date):
        df_train_num, df_train_cat = self.read_and_preprocess(path_train)
        df_test_num, df_test_cat = self.read_and_preprocess(path_test)
        self.targets_df = read_csv(targets_path, index_col=0)

        self.x_train_num, self.x_train_cat, self.y_train = self.split_features_labels(df_train_num, df_train_cat,
                                                                                      starting_date + 1)
        self.x_test_num, self.x_test_cat, self.y_test = self.split_features_labels(df_test_num, df_test_cat,
                                                                                   starting_date + 2)

        # Normalize the training and test data using StandardScaler for numerical data only
        scaler = StandardScaler()
        self.x_train_num = scaler.fit_transform(self.x_train_num)
        self.x_test_num = scaler.transform(self.x_test_num)

        # Calculate cardinalities and d_token
        self.cardinalities_train, self.cardinalities_test = self.calculate_cardinalities(df_train_cat, df_test_cat)

    def get_data(self):
        return {
            'train': (self.x_train_num, self.x_train_cat, self.y_train),
            'test': (self.x_test_num, self.x_test_cat, self.y_test),
        }

    def read_and_preprocess(self, path):
        df = read_csv(path, header=None)
        df = df.iloc[1:, 2:]

        # Separate categorical and numerical data
        df_cat = df.loc[:, df.nunique() == 2]  # consider binary and categorical columns
        df_num = df.loc[:, (df.nunique() > 2) | (df.nunique() == 1)]  # consider numeric columns

        return df_num, df_cat

    def split_features_labels(self, df_num, df_cat, target_date):
        x_num = df_num.values.astype('float32')
        x_cat = df_cat.values.astype('int64')  # categorical values are converted to int

        y = self.targets_df.values[:, target_date]
        y = LabelEncoder().fit_transform(y)
        y = y.astype('float32').reshape((len(y), 1))

        return x_num, x_cat, y

    def calculate_cardinalities(self, df_train_cat, df_test_cat):
        cardinalities_test = [df_test_cat[col].nunique() for col in df_test_cat.columns]
        cardinalities_train = [df_train_cat[col].nunique() for col in df_train_cat.columns]
        return cardinalities_train, cardinalities_test

    def __len__(self):
        return len(self.x_train_num)

    def __getitem__(self, idx):
        return [self.x_train_num[idx], self.x_train_cat[idx], self.y_train[idx]]
