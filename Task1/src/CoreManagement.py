import os, time, math
import random
import numpy as np
import torch
import torch.nn.functional as F

from .data_factory import DataFactory
from .log_factory import LogFactory
from .Normalizer import UnitGaussianNormalizer
from .models.nnet_model import MyNNet
from .utils import *
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV


class CoreComponent:
    def __init__(self, model='lasso', imputer='knn', outlier='zscore', pca='pca', device='cuda'):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.imputer = imputer
        self.outlier = outlier
        self.pca = pca
        self.device = device  # choose with your preference

        self.model_name = 'lasso' if model is None else model  # choose with your preference
        if self.model_name == 'lasso':
            self.train_model = LassoCV
        elif self.model_name == 'nnet':
            self.train_model = MyNNet(self)
        elif self.model_name == 'ridge':
            from sklearn.linear_model import RidgeCV
            self.train_model = RidgeCV
        else:
            self.train_model = None

        self.log_factory = LogFactory(self, log_to_disk=False)
        self.data_factory = DataFactory(self)
        self.full_normalizer = UnitGaussianNormalizer(self)
        self.y_normalizer = UnitGaussianNormalizer(self)

        self.full_X = None
        self.full_Y = None
        self.validation_X = None

        self.k_fold = 10
        self.train_percent = 0.99

        self.initialized = False

    def initialization(self):
        random.seed(0)
        self.log_factory.initialization()
        self.log_factory.InfoLog(sentences="Log Factory fully created")

        self.data_factory.initialization()

        # 1. read data
        self.full_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_train.csv"))
        self.full_Y = self.data_factory.read_dataset(os.path.join(self.data_path, "y_train.csv"))
        self.validation_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_test.csv"))

        # 2. process X files together
        full_X_shape_0 = self.full_X.shape[0]
        validation_X_shape_0 = self.validation_X.shape[0]
        full_validation_X = np.concatenate((self.full_X, self.validation_X), axis=0)

        full_validation_X, self.full_Y = self.data_factory.process_dataset(full_validation_X, self.full_Y,
                                                                           impute_method=self.imputer,
                                                                           outlier_method=self.outlier)
        self.full_normalizer.initialization(full_validation_X)
        full_validation_X = self.full_normalizer.encode(full_validation_X)
        full_X_shape_0 = len(self.full_Y)
        full_validation_X, self.full_Y = self.data_factory.feature_selection(full_validation_X, self.full_Y,
                                                                             method=self.pca, rows_X=full_X_shape_0)
        self.log_factory.InfoLog("After feature selection, the shape of X = {}".format(full_validation_X.shape))
        self.full_X = full_validation_X[:full_X_shape_0, :]
        self.validation_X = full_validation_X[-validation_X_shape_0:, :]

        # self.y_normalizer.initialization(self.full_Y)
        # self.full_Y = self.y_normalizer.encode(self.full_Y)

        # 3. transfer numpy data to Tensor data
        self.log_factory.InfoLog("Read data completed from X_train.csv, with shape as {}".format(self.full_X.shape))
        self.full_X = torch.autograd.Variable(torch.from_numpy(np.array(self.full_X)).float()).to(self.device)
        # self.full_Y = self.data_factory.process_dataset(self.full_Y) # Y data cannot be processed!
        self.log_factory.InfoLog("Read data completed from y_train.csv, with shape as {}".format(self.full_Y.shape))
        self.full_Y = torch.autograd.Variable(
            torch.from_numpy(np.array(self.full_Y).reshape(self.full_Y.shape[0], 1)).float()).to(self.device)

        self.log_factory.InfoLog(
            "Read data completed from X_test.csv, with shape as {}".format(self.validation_X.shape))
        self.validation_X = torch.autograd.Variable(torch.from_numpy(np.array(self.validation_X)).float()).to(
            self.device)

        self.initialized = True

    def run(self):
        if self.model_name == "lasso":
            full_X = self.full_X.cpu().numpy()
            full_Y = self.full_Y.cpu().numpy()
            reg = self.train_model(n_alphas=100, cv=self.k_fold, eps=1e-3, max_iter=5000, random_state=0,
                                   precompute=False).fit(full_X, full_Y)
            predicted_y_validate = reg.predict(self.validation_X.cpu().numpy())
            predicted_y_full = reg.predict(full_X)
            self.dump_validated_y(predicted_y_validate)
            self.log_factory.InfoLog("all score = {}".format(r2_score(full_Y, predicted_y_full)))
        elif self.model_name == 'ridge':
            full_X = self.full_X.cpu().numpy()
            full_Y = self.full_Y.cpu().numpy()
            """
            params: cv=k-fold //为None时使用loocv来验证，但是score会用mse而不是r2score
                    alphas=[...] //里面是我们备选的所有正则化参数
                    fit_intercept=True //default就是True，指在拟合时是否需要截距（当然需要）
            """
            reg = self.train_model(alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10.0], cv=self.k_fold).fit(full_X, full_Y)
            predicted_y_validate = reg.predict(self.validation_X.cpu().numpy())
            predicted_y_full = reg.predict(full_X)
            self.dump_validated_y(predicted_y_validate.squeeze(1))
            self.log_factory.InfoLog("all score = {}".format(r2_score(full_Y, predicted_y_full)))
        elif self.model_name == "mlp":
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
            from sklearn.neural_network import MLPRegressor

            row_idx = [i for i in range(self.full_X.shape[0])]
            random.shuffle(row_idx)
            train_X = self.full_X[0:math.floor(len(row_idx) * self.train_percent), ...].cpu().numpy()
            val_X = self.full_X[train_X.shape[0]:, ...].cpu().numpy()
            train_Y = self.full_Y[0:train_X.shape[0], ...].cpu().numpy()
            val_Y = self.full_Y[train_X.shape[0]:, ...].cpu().numpy()

            n_estimators = [int(x) for x in np.linspace(start=1000, stop=2000, num=10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num=5)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            print(random_grid)

            est = ExtraTreesRegressor()
            rf_random = RandomizedSearchCV(estimator=est, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
                                           n_jobs=-1)
            rf_random.fit(train_X, train_Y)
            rf_random.best_params_

            parameter_grid = {'learning_rate': ['constant', 'invscaling', 'adaptive']}

            start_time = time.time()

            regf_grid = GridSearchCV(MLPRegressor(max_iter=10000, activation='tanh', solver='sgd'), parameter_grid,
                                     n_jobs=-1)
            regf_grid.fit(train_X, train_Y)

            print('{} seconds'.format(round(time.time() - start_time)))
            print()
            print("Best parameters set found on development set:")
            print(regf_grid.best_params_)
            print()
            print("Grid scores on development set:")
            means = regf_grid.cv_results_['mean_test_score']
            stds = regf_grid.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, regf_grid.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            regf = ExtraTreesRegressor(n_estimators=1740, max_depth=53)
            regf_cv = cross_val_score(regf, train_X, train_Y, cv=5, n_jobs=-1)

            print("r2: %0.4f (+/- %0.2f)" % (regf_cv.mean(), regf_cv.std() * 2))

            # extra trees regression
            extra_tree = ExtraTreesRegressor(random_state=0, n_estimators=1740, max_depth=60, n_jobs=-1)
            extra_tree.fit(train_X, train_Y)
            extra_pred = extra_tree.predict(val_X)

            self.log_factory.InfoLog("The score of extra_tree for validation={}".format(r2_score(val_Y, extra_pred)))

            ID = np.array(range(len(val_X)))
            import pandas as pd
            df = pd.DataFrame({'id': ID,
                               'y': extra_pred})
            df.to_csv(os.path.join(self.data_path, 'prediction.csv'), index=False)
            self.dump_validated_y(extra_tree.predict(self.validation_X.cpu().numpy()))
        elif self.model_name == "nnet":
            self.train_model.initialization()
            computed_losses = []
            train_losses = []
            for epoch in range(self.train_model.total_epoch):
                stride = self.full_X.shape[0] // self.k_fold
                train_X, train_Y, test_X, test_Y = None, None, None, None

                test_loss = 0.0
                train_loss = 0.0
                test_mse = 0.0
                train_mse = 0.0
                test_r2_score = 0.0
                self.train_model.train()
                idx = [i for i in range(self.full_X.shape[0])]
                sampled_idx = random.sample(idx, self.full_Y.shape[0])
                for i in range(self.k_fold):
                    indicator = np.array([False for i in range(self.full_X.shape[0])])
                    if i != self.k_fold - 1:
                        indicator[sampled_idx[i * stride: (i + 1) * stride]] = True
                    else:
                        indicator[sampled_idx[i * stride:]] = True
                    # k-fold CV
                    train_X = self.full_X[indicator == False, :]
                    train_Y = self.full_Y[indicator == False, :]
                    test_X = self.full_X[indicator == True, :]
                    test_Y = self.full_Y[indicator == True, :]

                    self.train_model.optimizer.zero_grad()
                    predicted_y = self.train_model(train_X)
                    temp_loss = self.train_model.compute_loss(predicted_y, train_Y)
                    temp_loss.backward()
                    self.train_model.optimizer.step()

                    with torch.no_grad():
                        train_loss += temp_loss.item() / self.k_fold
                        predicted_y_test = self.train_model(test_X)
                        test_loss += self.train_model.compute_loss(predicted_y_test, test_Y) / self.k_fold
                        train_mse += F.mse_loss(predicted_y, train_Y) / self.k_fold
                        test_mse += F.mse_loss(predicted_y_test, test_Y) / self.k_fold
                        test_r2_score += r2_score(test_Y.cpu().numpy(), predicted_y_test.cpu().numpy()) / self.k_fold

                if epoch % 200 == 0:
                    self.log_factory.InfoLog(
                        "Epoch={}, while test loss={}, train loss={}, test MSE={}, train MSE={}, r2_score={}".format(
                            epoch, test_loss, train_loss, test_mse, train_mse, test_r2_score))
                    computed_losses.append(test_loss.detach().clone().cpu())
                    train_losses.append(train_loss)
                    with torch.no_grad():
                        predicted_y_validate = self.train_model(self.validation_X).squeeze(1).cpu().numpy()
                        self.dump_validated_y(predicted_y_validate)
            model_evaluation(computed_losses, train_losses, epoch_step=200)

    def kill(self):
        self.log_factory.kill()

    def dump_validated_y(self, predicted_y_validate):
        np_full_Y = self.full_Y
        try:
            np_full_Y = self.full_Y.squeeze(1).cpu().numpy()
            predicted_y_validate = predicted_y_validate.cpu().numpy()
        except:
            pass

        if self.y_normalizer.initialized:
            predicted_y_validate = self.y_normalizer.decode(predicted_y_validate)
            np_full_Y = self.y_normalizer.decode(np_full_Y)

        fig = plt.figure(1)
        plt.scatter([1 for i in range(self.full_Y.shape[0])], np_full_Y, edgecolors='r')
        plt.scatter([2 for i in range(len(predicted_y_validate))], predicted_y_validate, edgecolors='b')
        fig.savefig(os.path.join(self.data_path, "distribution.png"))

        with open(os.path.join(self.data_path, "y_validate.csv"), 'w') as f:
            f.write("id,y\n")
            for i, pred_y in enumerate(predicted_y_validate):
                f.write("{},{}\n".format(i, pred_y))
            f.close()
