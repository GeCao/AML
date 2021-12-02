from math import nan
import os, time, math
import random
import torch

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator, IterativeImputer
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn import feature_selection
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, RFECV
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


class DataFactory:
    def __init__(self, core_management):
        self.core_management = core_management

        self.anomalies_ratio = 0.25  # 预测样本中outlier的比例
        self.max_features = 200

        self.train_dataset_filter = None  # to filter y_train when x_train is trimmed
        self.nan_indicator_mat = None

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def winsorize(self, X):
        df = pd.DataFrame(X.T)
        # 将因子值进行极端值缩尾处理，拉回至3.5倍MAD水平，并且不影响排序，不减少覆盖度
        md = df.median(axis=1)
        mad = (1.483 * (df.sub(md, axis=0)).abs().median(axis=1)).replace(0, np.nan)
        up = df.apply(lambda k: k > md + mad * 3)
        down = df.apply(lambda k: k < md - mad * 3)
        df[up] = df[up].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md + mad * 3, axis=0)
        df[down] = df[down].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md - mad * (0.5 + 3), axis=0)
        return df.T

    def outlier_detect(self, X, y=None, method='zscore'):
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
            print('2')
            rows_X = y.shape[0]
            X_y = np.concatenate((X[:rows_X, ...], y.reshape((rows_X, 1))), axis=1)
            if_sk = IsolationForest(n_estimators=100,
                                    max_samples="auto",
                                    contamination="auto",
                                    random_state=np.random.RandomState(42))
            if_sk.fit(X_y)
            if_predict = if_sk.predict(X_y)
            if_filter = if_predict > 0  # map +1/-1 to True/False
            if self.nan_indicator_mat is not None:
                self.nan_indicator_mat = np.concatenate((self.nan_indicator_mat[:rows_X, ...][if_filter],
                                                         self.nan_indicator_mat[rows_X:, ...]),
                                                        axis=0)  # this part is readied for future impute

            if_X_y = X_y[:rows_X, ...][if_filter]
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
        elif method == 'winsorize':
            X = self.winsorize(X).to_numpy()
            y = self.winsorize(y).to_numpy().astype(int)
        else:
            X, y = X, y
        return X, y

    def feature_selection(self, X, y, method='rfecv'):
        rows_X = y.shape[0]
        train_X = X[:rows_X, ...]

        if (method == 'pca') & (X.shape[1] > 2):
            estimator = PCA(n_components=256)
            df = estimator.fit_transform(X)
            return df, y
        elif method == 'rfecv':
            rfecv = RFECV(estimator=ExtraTreesRegressor(n_estimators=1470, n_jobs=-1), step=1, cv=KFold(2), n_jobs=-1)
            train_X = X[:rows_X, ...]
            rfecv = rfecv.fit(train_X, y)
            X = rfecv.transform(X)
            self.nan_indicator_mat = rfecv.transform(self.nan_indicator_mat)
            print("Optimal number of features : %d" % rfecv.n_features_)
            return X, y
        elif method == 'tree':
            clf = ExtraTreesClassifier(n_estimators=50)
            clf = clf.fit(train_X, y.astype('int'))
            model = SelectFromModel(clf, prefit=True)
            X = model.transform(X)
            self.nan_indicator_mat = model.transform(self.nan_indicator_mat)
            return X, y
        elif method == 'lsvc':
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_X, y.astype('int'))
            print(lsvc.score(train_X, y))
            model = SelectFromModel(lsvc, prefit=True, threshold=-np.inf, max_features=self.max_features)
            X = model.transform(X)
            self.nan_indicator_mat = model.transform(self.nan_indicator_mat)
            return X, y
        elif method == 'lassoCV':
            lasso = LassoCV(tol=9.5, max_iter=10000).fit(train_X, y.ravel())
            importance = np.abs(lasso.coef_)
            threshold = min(i for i in importance if i > 0) - 0.0001
            print(lasso.score(train_X, y))
            model = SelectFromModel(lasso, threshold=threshold).fit(train_X, y.ravel())
            X = model.transform(X)
            self.nan_indicator_mat = model.transform(self.nan_indicator_mat)
            return X, y
        elif method == 'lasso':
            lasso = Lasso(tol=9.5, max_iter=10000, alpha=0.02).fit(train_X, y.ravel())
            model = SelectFromModel(lasso, prefit=True, threshold=-np.inf, max_features=self.max_features)
            print('lasso_score:', lasso.score(train_X, y))
            X = model.transform(X)
            self.nan_indicator_mat = model.transform(self.nan_indicator_mat)
            return X, y
        elif method == 'SelectPercentile':
            # 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化
            percentiles = range(1, 100, 2)
            results = []

            # extra trees regression
            extra_tree = ExtraTreesRegressor(n_estimators=1189, max_depth=62, max_features='auto',
                                             min_samples_leaf=1, min_samples_split=2, n_jobs=-1, bootstrap=True)

            for i in percentiles:
                train_X = MinMaxScaler().fit_transform(train_X)
                fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)  # chi2卡方检验
                X_train_fs = fs.fit_transform(train_X, y)
                val = cross_val_score(extra_tree, X_train_fs, y, scoring='r2',
                                      cv=5, n_jobs=-1).mean()
                results = np.append(results, val)

            print('res_max', results.max())
            opt = np.where(results == results.max())[0]

            opt_percentiles = percentiles[int(opt)]
            print('opt_percent', percentiles[int(opt)])

            # 作图
            import pylab as pl
            pl.plot(percentiles, results)
            pl.xlabel('percentiles of features')
            pl.ylabel('accuracy')
            pl.show()

            fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=opt_percentiles)
            X_train_fs = fs.fit_transform(train_X, y)
            X_fs = fs.transform(X)
            return X_fs, y
        else:
            return X, y

    def filter_low_var_feature(self, x, threshold=1e-10):
        x = pd.DataFrame(x)
        x += 1e-7
        x_var = np.var(x / np.mean(x, axis=0), axis=0)
        x_idx = x_var > threshold
        x -= 1e-7
        return x_idx

    def impute(self, X, y, method='else'):
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
            lr = LinearRegression()
            mice_imputer = IterativeImputer(estimator=lr, verbose=2, max_iter=30, tol=1e-4, imputation_order='roman')
            X = mice_imputer.fit_transform(X)
            y = mice_imputer.fit_transform(y)
            return X, y
        elif method == 'mean' or method == 'median':
            # normalization
            std = StandardScaler()
            idx = self.filter_low_var_feature(X, 1e-10)
            if self.nan_indicator_mat is not None:
                self.nan_indicator_mat = self.nan_indicator_mat[:, idx]
            X = X[:, idx]
            X = std.fit_transform(X)

            mean_imputer = SimpleImputer(missing_values=np.nan, strategy=method)
            X = mean_imputer.fit_transform(X)
            y = mean_imputer.fit_transform(y)
            return X, y
        elif method == 'random_forest':
            data_copy = pd.DataFrame(X).copy()

            sindex = np.argsort(data_copy.isna().sum().values.tolist())  # 将有缺失值的列按缺失值的多少由小到大
            print(sindex)
            # 进入for循环进行空值填补
            for i in sindex:  # 按空值数量,从小到大进行排序来遍历
                print(i)
                if data_copy.iloc[:, i].isna().sum() == 0:  # 将没有空值的行过滤掉
                    continue  # 直接跳过当前的for循环
                df = data_copy  # 复制df数据
                fillc = df.iloc[:, i]  # 将第i列的取出，之后作为y变量
                df = df.iloc[:, df.columns != df.columns[i]]  # 除了有这列以外的数据，之后作为X
                df_0 = SimpleImputer(missing_values=np.nan,  # 将df的数据全部用0填充
                                     strategy="constant",
                                     fill_value=0).fit_transform(df)
                Ytrain = fillc[fillc.notnull()]  # 在fillc列中,不为NAN的作为Y_train
                Ytest = fillc[fillc.isnull()]  # 在fillc列中,为NAN的作为Y_test
                Xtrain = df_0[Ytrain.index, :]  # 在df_0中(已经填充了0),中那些fillc列不为NAN的行作为Xtrain
                Xtest = df_0[Ytest.index, :]  # 在df_0中(已经填充了0),中那些fillc等于NAN的行作为X_test

                rfc = RandomForestRegressor()
                rfc.fit(Xtrain, Ytrain)
                Ypredict = rfc.predict(Xtest)  # Ytest为了定Xtest,以最后预测出Ypredict

                data_copy.loc[data_copy.iloc[:, i].isnull(), data_copy.columns[i]] = Ypredict
                # 将data_copy中data_copy在第i列为空值的行,第i列,改成Ypredict    
            X = data_copy.to_numpy()
            # 填充y值
            mean_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            y = mean_imputer.fit_transform(y)
            return X, y
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

        data = pd.read_csv(file_path, dtype=np.float32)
        data = data.dropna(axis=1, how='any')
        column_name_list = data.columns.tolist()
        if 'id' in column_name_list:
            # data['id'] = data['id'].astype(int)
            del data['id']

        return data.to_numpy()

    def select_K(self, X, y):
        train_validation_split_idx = y.shape[0]
        model = SelectKBest(score_func=f_regression, k=self.max_features)
        model.fit(X[:train_validation_split_idx, ...], y)
        X = model.transform(X)
        self.nan_indicator_mat = model.transform(self.nan_indicator_mat)
        return X, y

    def preprocess_dataset(self, X, y):
        missing_indicator = MissingIndicator(missing_values=nan, features='all')
        self.nan_indicator_mat = missing_indicator.fit_transform(X)  # 得到一个与data同shape的矩阵，True代表有nan，False代表正常数据

        X, y = self.impute(X, y, 'median')  # knn对outlier非常敏感，因此这只是一个为了outlier detection做出的预处理
        self.core_management.full_normalizer.initialization(X)
        X = self.core_management.full_normalizer.encode(X)
        X, y = self.select_K(X, y)
        return X, y

    def impute_dataset(self, X, y, impute_method='knn'):
        X = self.original_nan_mask(X)

        X, y = self.impute(X, y, impute_method)  # knn对outlier非常敏感，因此这只是一个为了outlier detection做出的预处理
        return X, y

    def outlier_detect_dataset(self, X, y, outlier_method='isolationforest'):
        X, y = self.outlier_detect(X, y, outlier_method)
        return X, y

    def original_nan_mask(self, X):
        X = np.where(self.nan_indicator_mat, nan, X)  # 此前用knn做的impute非常差，所以我们要重新mask掉，然后做最终的impute
        return X
