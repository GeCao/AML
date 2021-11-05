from math import nan
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


class DataFactory:
    train_dataset_filter = None 
    # to filter y_train when x_train is trimmed 

    def __init__(self, core_management):
        self.core_management = core_management

        self.anomalies_ratio = 0.2  # 预测样本中outlier的比例
        self.max_features = 200

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def outlier_detect_data(self, df_data, method='zscore', rows_X=0):
        # make these outlier entries nan
        if method == 'zscore': 
            z_scores = stats.zscore(df_data)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores > 3)
            df_data[filtered_entries] = nan
            new_df = df_data 
            # new_df = df_data[filtered_entries]
        elif method == 'isolationforest':
            if_sk = IsolationForest(n_estimators=100,
                                    max_samples="auto",
                                    contamination=self.anomalies_ratio,
                                    random_state=np.random.RandomState(42))
            if_sk.fit(df_data)
            if_predict = if_sk.predict(df_data)
            if_filter = if_predict > 0  # map +1/-1 to True/False
            if_filter = if_filter[:rows_X] 
            self.train_dataset_filter = if_filter 
            
            if_df = df_data[:rows_X][if_filter] 
            zs_df = self.outlier_detect_data(df_data[rows_X:], 'zscore') 
            new_df = np.concatenate((if_df, zs_df), axis=0)
        else:
            new_df = df_data
        return new_df

    def feature_selection(self, X, y, method='pca', rows_X=None):
        train_X = X
        """
        解释一下rows_X这个参数，因为现在train_X和test_X被合并到了一起，所以我们希望填入train_X的行数来把train_X提取出来
        """
        if rows_X is not None:
            train_X = X[:rows_X, ...]

        if (method == 'pca') & (X.shape[1] > 2):
            estimator = PCA(n_components=256)
            df = estimator.fit_transform(X)
            return df, y
        elif method == 'tree':
            clf = ExtraTreesClassifier(n_estimators=50)
            clf = clf.fit(train_X, y.astype('int'))
            model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=self.max_features)
            X = model.transform(X)
            return X, y
        elif method == 'lsvc':
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_X, y.astype('int'))
            print(lsvc.score(train_X, y))
            model = SelectFromModel(lsvc, prefit=True, threshold=-np.inf, max_features=self.max_features)
            X = model.transform(X)
            return X, y
        elif method == 'lasso':
            lasso = LassoCV(tol=9.5, max_iter=10000).fit(train_X, y.ravel())
            importance = np.abs(lasso.coef_)
            threshold = min(i for i in importance if i >0)-0.0001
            print(lasso.score(train_X, y))
            model = SelectFromModel(lasso, threshold=threshold).fit(train_X,y.ravel())
            X = model.transform(X)
            return X, y
        else:
            return X, y

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
            """
            column_name_list = df_data.columns.tolist()
            for col_name in column_name_list:
                if df_data[col_name].isna().any():
                    nan_part = df_data[df_data[col_name].isna()]
                    real_part = df_data[col_name].dropna()
                    # model = KNeighborsRegressor(n_neighbors=5).fit()
                    return df_data
            """
            knn_imputer = KNNImputer(n_neighbors=5)
            df_data = knn_imputer.fit_transform(df_data)
            return df_data
        elif method == 'delete':
            df_data = df_data.dropna(axis=0, how='any')  # Delete this row as long as a nan has been detected
            return df_data
        elif method == 'mean':
            df_data = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
                df_data).transform(df_data)
            return df_data
        else:
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

        return data.to_numpy()

    def process_dataset(self, data, impute_method='knn', outlier_method='zscore', pca_method='pca', rows_X=0):
        # read_dataset() must be followed by the process_dataset()
        # rows_X to help not to delete outliers of x_test
        if self.train_dataset_filter is not None:
            # trim Y according to X
            data = data[self.train_dataset_filter]
        data = self.impute_data(data, impute_method)
        data = self.outlier_detect_data(data, outlier_method, rows_X)
        data = self.impute_data(data, impute_method)

        try:
            data = data.to_numpy()
        except:
            pass
        return data 
