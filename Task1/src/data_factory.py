from math import nan
import os
import numpy as np
import pandas as pd
from scipy import stats
from .data_utils.gain_imputer import GAINImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator, IterativeImputer
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, RFECV
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV


class DataFactory:
    def __init__(self, core_management):
        self.core_management = core_management

        self.anomalies_ratio = 0.25  # 预测样本中outlier的比例
        self.max_features = 200

        self.gain_imputer = GAINImputer(self.core_management)

        self.train_dataset_filter = None  # to filter y_train when x_train is trimmed
        self.nan_indicator_mat = None

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def outlier_detect_data(self, X, y=None, method='zscore'):
        """
        There are only outliers in X_train & y_train dataset!
        :param X:
        :param y:
        :param method:
        :return:
        """
        # make these outlier entries nan
        if method == 'zscore':
            z_scores = stats.zscore(X)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores > 3)
            X[filtered_entries] = nan
            if y is None:
                return X, y

            z_scores = stats.zscore(y)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores > 3)
            y[filtered_entries] = nan
        elif method == 'local':
            rows_X = y.shape[0]
            X_y = np.concatenate((X[:rows_X, ...], y.reshape((rows_X, 1))), axis=1)
            model = LocalOutlierFactor(novelty=True)
            model.fit(X_y)
            if_predict = model.predict(X_y)
            if_filter = if_predict > 0  # map +1/-1 to True/False
            if self.nan_indicator_mat is not None:
                self.nan_indicator_mat = np.concatenate((self.nan_indicator_mat[:rows_X, ...][if_filter],
                                                         self.nan_indicator_mat[rows_X:, ...]),
                                                        axis=0)  # this part is readied for future impute

            if_X_y = X_y[:rows_X][if_filter]
            X = np.concatenate((if_X_y[:, :-1], X[rows_X:]), axis=0)
            y = if_X_y[:, -1]
        elif method == 'isolationforest':
            rows_X = y.shape[0]
            X_y = np.concatenate((X[:rows_X, ...], y.reshape((rows_X, 1))), axis=1)
            if_sk = IsolationForest(n_estimators=100,
                                    max_samples="auto",
                                    contamination=self.anomalies_ratio,
                                    random_state=np.random.RandomState(42))
            if_sk.fit(X_y)
            if_predict = if_sk.predict(X_y)
            if_filter = if_predict > 0  # map +1/-1 to True/False
            if self.nan_indicator_mat is not None:
                self.nan_indicator_mat = np.concatenate((self.nan_indicator_mat[:rows_X, ...][if_filter],
                                                         self.nan_indicator_mat[rows_X:, ...]),
                                                        axis=0)  # this part is readied for future impute

            if_X_y = X_y[:rows_X][if_filter]
            X = np.concatenate((if_X_y[:, :-1], X[rows_X:, :]), axis=0)
            y = if_X_y[:, -1]
        elif method == 'iqr':
            rows_X = y.shape[0]
            X_y = np.concatenate((X[:rows_X, ...], y.reshape((rows_X, 1))), axis=1)
            Q1 = np.quantile(X_y, 0.25)  # X_y.quantile(0.25)
            Q3 = np.quantile(X_y, 0.75)  # X_y.quantile(0.75)
            IQR = Q3 - Q1
            X_y_no = X_y[~((X_y < (Q1 - 100 * IQR)) | (X_y > (Q3 + 100 * IQR))).any(axis=1)]
            X = np.concatenate((X_y_no[:, :-1], X[rows_X:, :]), axis=0)
            y = X_y_no[:, -1]
        else:
            X, y = X, y
        return X, y

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
        elif method == 'kbest':
            model = SelectKBest(score_func=f_regression, k=self.max_features)
            model.fit(train_X, y)
            X = model.transform(X)

            rfecv = RFECV(estimator=ExtraTreesRegressor(n_estimators=1470, n_jobs=-1), step=1, cv=KFold(2), n_jobs=-1)
            train_X = X[:rows_X, ...]
            rfecv = rfecv.fit(train_X, y)
            X = rfecv.transform(X)
            print("Optimal number of features : %d" % rfecv.n_features_)
            return X, y
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

    def impute_data(self, X, y, method='else'):
        """

        :param np_data: our data with DataFrame mode
        :param method: several choices:
                       1. 'knn': instead of 'nan' values in every column, we predict them with a simple knn algorithm,
                                 the train dataset comes from all the other non-nan data in this column
                       2. else: use the internal implementation of sklearn, choose 'mean' or 'median' strategy for predict.
        :return:
        """
        if method == 'knn':
            knn_imputer = KNNImputer(n_neighbors=5)
            X = knn_imputer.fit_transform(X)
            y = knn_imputer.fit_transform(y.reshape(y.shape[0], 1))
            return X, y
        elif method == 'mice':
            mice_imputer = IterativeImputer(max_iter=10, random_state=0)
            X = mice_imputer.fit_transform(X)
            y = mice_imputer.fit_transform(y)
            return X, y
        elif method == 'mean' or method == 'median':
            mean_imputer = SimpleImputer(missing_values=np.nan, strategy=method)
            X = mean_imputer.fit_transform(X)
            y = mean_imputer.fit_transform(y)
            return X, y
        elif method == 'gain':
            self.gain_imputer.initialization(X)
            return self.gain_imputer.train(), KNNImputer(n_neighbors=5).fit_transform(y.reshape(y.shape[0], 1))
        else:
            return X, y

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

    def process_dataset(self, X, y, impute_method='knn', outlier_method='zscore'):
        missing_indicator = MissingIndicator(missing_values=nan, features='all')
        self.nan_indicator_mat = missing_indicator.fit_transform(X)  # 得到一个与data同shape的矩阵，True代表有nan，False代表正常数据

        X, y = self.impute_data(X, y, impute_method)  # knn对outlier非常敏感，因此这只是一个为了outlier detection做出的预处理
        X, y = self.outlier_detect_data(X, y, outlier_method)
        # X = np.where(self.nan_indicator_mat, nan, X)  # 此前用knn做的impute非常差，所以我们要重新mask掉，然后做最终的impute
        # X, y = self.impute_data(X, y, impute_method)
        return X, y
