import os, time, math
import random
import numpy as np
import pandas as pd

from .data_factory import DataFactory
from .log_factory import LogFactory
from .Normalizer import UnitGaussianNormalizer
from .utils import *
from sklearn.metrics import f1_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LassoCV
from bayes_opt import BayesianOptimization
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV

import matplotlib.pyplot as plt


class CoreComponent:
    def __init__(self, model='lasso', imputer='knn', outlier='zscore', pca='pca', device='cuda'):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.imputer = imputer
        self.outlier = outlier
        self.pca = pca
        self.device = device  # choose with your preference

        self.model_name = 'svm' if model is None else model  # choose with your preference
        if self.model_name == 'svm':
            self.train_model = LassoCV
        else:
            self.train_model = None

        self.log_factory = LogFactory(self, log_to_disk=False)
        self.data_factory = DataFactory(self)
        self.full_normalizer = UnitGaussianNormalizer(self)

        self.full_X = None
        self.full_Y = None
        self.validation_X = None

        self.k_fold = 10
        self.train_percent = 0.8

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
        # full_validation_X, self.full_Y = self.data_factory.post_impute(full_validation_X, self.full_Y, impute_method=self.imputer)
        self.full_X = full_validation_X[:full_X_shape_0, :]
        self.validation_X = full_validation_X[-validation_X_shape_0:, :]

        self.log_factory.InfoLog("Read data completed from X_train.csv, with shape as {}".format(self.full_X.shape))
        self.log_factory.InfoLog("Read data completed from y_train.csv, with shape as {}".format(self.full_Y.shape))
        self.log_factory.InfoLog("Read data completed from X_test.csv, with shape as {}".format(self.validation_X.shape))

        self.full_Y = self.full_Y.astype(np.int)
        cls0, cls1, cls2, cls3 = np.bincount(self.full_Y.reshape(-1))
        total = cls0 + cls1 + cls2 + cls3
        print('Samples:\n    Total: {}\n \
              Class 0: {} ({:.2f}% of total)\n \
              Class 1: {} ({:.2f}% of total)\n \
              Class 2: {} ({:.2f}% of total)\n \
              Class 3: {} ({:.2f}% of total)'.format(total, cls0, 100 * cls0 / total, cls1, 100 * cls1 / total, cls2,
                                                     100 * cls2 / total, cls3, 100 * cls3 / total))

        self.initialized = True

    def run(self):
        if self.model_name == "svm":
            plot_stats(self.full_X)

            cls0_bool = self.full_Y == 0
            cls1_bool = self.full_Y == 1
            cls2_bool = self.full_Y == 2
            cls3_bool = self.full_Y == 3

            cls0_X = self.full_X[cls0_bool.reshape(-1), :]
            cls1_X = self.full_X[cls1_bool.reshape(-1), :]
            cls2_X = self.full_X[cls2_bool.reshape(-1), :]
            cls3_X = self.full_X[cls3_bool.reshape(-1), :]

            plot_classes(cls0_X, cls1_X, cls2_X, cls3_X)

            C = [0.01, 0.1, 0.4, 0.8]
            kernel = ['linear', 'rbf', 'poly', 'sigmoid']
            gamma = [0.1, 0.2, 0.4]
            degree = [0, 1, 2, 3, 4, 5, 6]
            tol = [0.1, 0.01, 0.001, 0.0001]
            decision_function_shape = ['ovr', 'ovo']

            random_grid = {'C': C,
                           'kernel': kernel,
                           'gamma': gamma,
                           'degree': degree,
                           'tol': tol,
                           'decision_function_shape': decision_function_shape}

            seed = 5
            est = svm.SVC(class_weight='balanced', random_state=seed)
            search = RandomizedSearchCV(estimator=est,
                                        param_distributions=random_grid,
                                        n_iter=100,
                                        cv=3,
                                        verbose=1,
                                        n_jobs=-1,
                                        scoring='balanced_accuracy')
            search.fit(self.full_X, self.full_Y)
            search.best_params_

            search.cv_results_

            search_weights = [{0: 3.2, 1: 0.44444444, 2: 3.05},
                              {0: 3.05, 1: 0.44444444, 2: 3.2}]
            parameter_grid = {'class_weight': search_weights}

            seed = 77
            est = svm.SVC(random_state=seed,
                          kernel='rbf',
                          tol=0.001,
                          gamma=0.057,
                          degree=6,
                          decision_function_shape='ovo',
                          C=0.6225)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            tuning = GridSearchCV(est, parameter_grid, cv=skf, n_jobs=-1, scoring='balanced_accuracy')
            tuning.fit(self.full_X, self.full_Y)

            self.log_factory.InfoLog("Best parameters set found on development set:")
            self.log_factory.InfoLog("{}".format(tuning.best_score_))
            self.log_factory.InfoLog("{}".format(tuning.best_params_))
            self.log_factory.InfoLog("Grid scores on development set:")
            means = tuning.cv_results_['mean_test_score']
            stds = tuning.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, tuning.cv_results_['params']):
                self.log_factory.InfoLog("%0.6f (+/-%0.04f) for %r" % (mean, std * 2, params))

            tuning.best_estimator_
            best_est = tuning.best_estimator_

            weights = {0: 3.18, 1: 0.44444444, 2: 3.025}
            seed = 77
            svmc = svm.SVC(class_weight=weights,
                           random_state=seed,
                           kernel='rbf',
                           tol=0.001,
                           gamma=0.057,
                           degree=6,
                           decision_function_shape='ovo',
                           C=0.6225)
            seed = 77
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            score = []
            for train_index, val_index in skf.split(self.full_X, self.full_Y):
                X_train, X_val = self.full_X[train_index], self.full_X[val_index]
                y_train, y_val = self.full_Y[train_index], self.full_Y[val_index]

                # train model
                svmc.fit(X_train, y_train)
                # best_est.fit(X_train, y_train)
                prediction = svmc.predict(X_val)
                # prediction = best_est.predict(X_val)
                BMAC = balanced_accuracy_score(y_val, prediction)
                score.append(BMAC)
                self.log_factory.InfoLog('BMAC: {:0.5f}'.format(BMAC))
            self.log_factory.InfoLog('CV complete.')
            score = np.array(score)
            self.log_factory.InfoLog("BMAC: %0.4f (+/- %0.4f)" % (np.mean(score), np.std(score) * 2))

            score = list()

            svmc.fit(self.full_X, self.full_Y)

            plot_confusion_matrix(svmc, self.full_X, self.full_Y)

            svmc.fit(self.full_X, self.full_Y)

            tuned_svc = svmc.predict(self.validation_X)

            ID = np.array(range(len(self.validation_X)))
            df = pd.DataFrame({'id': ID,
                               'y': tuned_svc})
            name = 'y_validation.csv'
            path = os.path.join('.', name)
            df.to_csv(path, index=False)

            # self.dump_validated_y(predicted_y_validate)
            # self.log_factory.InfoLog("all score = {}".format(f1_score(full_Y, predicted_y_full, average='micro')))


    def kill(self):
        self.log_factory.kill()

    def dump_validated_y(self, predicted_y_validate):
        np_full_Y = self.full_Y
        try:
            np_full_Y = self.full_Y.squeeze(1).cpu().numpy()
            predicted_y_validate = predicted_y_validate.cpu().numpy()
        except:
            pass

        fig = plt.figure(1)
        plt.scatter([1 for i in range(self.full_Y.shape[0])], np_full_Y, edgecolors='r')
        plt.scatter([2 for i in range(len(predicted_y_validate))], predicted_y_validate, edgecolors='b')
        fig.savefig(os.path.join(self.data_path, "distribution.png"))

        with open(os.path.join(self.data_path, "y_validate.csv"), 'w') as f:
            f.write("id,y\n")
            for i, pred_y in enumerate(predicted_y_validate):
                f.write("{},{}\n".format(i, pred_y))
            f.close()
