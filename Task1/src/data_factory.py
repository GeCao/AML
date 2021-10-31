import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


class DataFactory:
    def __init__(self, core_management):
        self.core_management = core_management
        self.initialized = False

    def initialization(self):
        self.initialized = True

    def outlier_detect_data(self, df_data, method='zscore'):
        if method == 'zscore': 
            z_scores = stats.zscore(df_data)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            new_df = df_data[filtered_entries]
        else:
            new_df = df_data
        return new_df

    def PCA_data(self, df_data, method='else'):
        return df_data

    def impute_data(self, df_data, method='else'):
        """

        :param df_data: our data with DataFrame mode
        :param method: several choices:
                       1. 'knn': instead of 'nan' values in every column, we predict them with a simple knn algorithm,
                                 the train dataset comes from all the other non-nan data in this column
                       2. 'delete': delete every row as long as it got a nan number in any column,
                                    this is a rather dummy way for data cleaning
                       3. else: use the internal implementation of sklearn, choose 'mean' or 'median' strategy for predict.
        :return:
        """
        if method == 'knn':
            # TODO: this method has not been fully implemented
            column_name_list = df_data.columns.tolist()
            for col_name in column_name_list:
                if df_data[col_name].isna().any():
                    nan_part = df_data[df_data[col_name].isna()]
                    real_part = df_data[col_name].dropna()
                    # model = KNeighborsRegressor(n_neighbors=5).fit()
                    return df_data
        elif method == 'delete':
            df_data = df_data.dropna(axis=0, how='any')  # Delete this row as long as a nan has been detected
            return df_data
        else:
            arr = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
                df_data.values).transform(df_data.values)
            df_data = pd.DataFrame(data=arr, index=df_data.index.values, columns=df_data.columns.values)
            return df_data

    def read_dataset(self, file_path):
        """

        :param file_path:
        :return: data

        Please note while 'id' column exists, we will always delete this column
        """
        if not os.path.exists(file_path):
            return None

        data = pd.read_csv(file_path)
        column_name_list = data.columns.tolist()
        if 'id' in column_name_list:
            # data['id'] = data['id'].astype(int)
            del data['id']

        return data

    def process_dataset(self, data, impute_method='mean', outlier_method='else', pca_method='else'):
        # read_dataset() must be followed by the process_dataset() 
        data = self.impute_data(data, impute_method)
        data = self.outlier_detect_data(data, outlier_method)
        data = self.PCA_data(data, pca_method)

        data = data.to_numpy()
        return data 
