import os, time, math
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg

from .data_factory import DataFactory
from .log_factory import LogFactory
from .Normalizer import UnitGaussianNormalizer
from .models.nnet_model import MyNNet
from .utils import *
from sklearn.metrics import f1_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeClassifier
from bayes_opt import BayesianOptimization
from sklearn import svm
from sklearn.svm import SVC
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier

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
        elif self.model_name == 'nnet':
            self.train_model = MyNNet(self)
        else:
            self.train_model = None

        self.log_factory = LogFactory(self, log_to_disk=False)
        self.data_factory = DataFactory(self)
        self.full_normalizer = UnitGaussianNormalizer(self)

        self.full_X = None
        self.full_Y = None
        self.validation_X = None

        self.k_fold = 10
        self.train_percent = 0.95

        self.initialized = False

    def initialization(self):
        random.seed(0)
        self.log_factory.initialization()
        self.log_factory.InfoLog(sentences="Log Factory fully created")

        self.data_factory.initialization()

        # 1. read data
        self.full_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_train.csv"), "X_train")
        self.full_Y = self.data_factory.read_dataset(os.path.join(self.data_path, "y_train.csv"), "y_train")
        self.validation_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_test.csv"), "X_test")
        self.validation_X = self.validation_X[:, :self.full_X.shape[1]]
        self.log_factory.InfoLog("Read data completed from X_train.csv, with shape as {}".format(self.full_X.shape))
        self.log_factory.InfoLog("Read data completed from y_train.csv, with shape as {}".format(self.full_Y.shape))
        self.log_factory.InfoLog("Read data completed from X_test.csv, with shape as {}".format(self.validation_X.shape))

        print("The mean of dataset of train_X: = {}".format(self.full_X.mean()))

        # 2. process X files together
        full_X_shape_0 = self.full_X.shape[0]
        validation_X_shape_0 = self.validation_X.shape[0]
        full_validation_X = np.concatenate((self.full_X, self.validation_X), axis=0)

        # 3. preprocess: impute with median + normalize + k_best
        # full_validation_X, self.full_Y = self.data_factory.preprocess_dataset(full_validation_X, self.full_Y)

        # 4. feature selection
        full_X_shape_0 = len(self.full_Y)
        # full_validation_X, self.full_Y = self.data_factory.feature_selection(full_validation_X, self.full_Y, method=self.pca)

        # 5. imputer
        # full_validation_X, self.full_Y = self.data_factory.impute_dataset(full_validation_X, self.full_Y, impute_method=self.imputer)

        # 6. outlier detection
        # full_validation_X, self.full_Y = self.data_factory.outlier_detect_dataset(full_validation_X, self.full_Y, outlier_method=self.outlier)

        meanhr = []
        minhr = []
        maxhr = []
        stdhr = []
        IBI = []
        SDNN = []
        SDSD = []
        RMSSD = []
        pNN20 = []
        pNN50 = []
        measures = {}

        def calc_RR(out):
            peaklist = out[0][out[2]]
            RR_interval = np.diff(peaklist) * 1000
            measures['ibi'] = np.mean(RR_interval)
            measures['sdnn'] = np.std(RR_interval)
            interval_diff = abs(np.diff(RR_interval))
            nn20 = [x for x in interval_diff if (x > 20)]
            nn50 = [x for x in interval_diff if (x > 50)]
            measures['pnn20'] = float(len(nn20)) / float(len(interval_diff)) if len(interval_diff) > 0 else np.nan
            measures['pnn50'] = float(len(nn50)) / float(len(interval_diff)) if len(interval_diff) > 0 else np.nan
            measures['rmssd'] = np.sqrt(np.mean(np.power(interval_diff, 2))) if len(interval_diff) > 0 else np.nan
            measures['sdsd'] = np.std(interval_diff) if len(interval_diff) > 0 else np.nan

        for i in range(full_validation_X.shape[0]):
            out = ecg.ecg(signal=full_validation_X[i][10:], sampling_rate=300, show=False)
            try:
                meanhr.append(np.mean(out[6]))
                minhr.append(min(out[6]))
                maxhr.append(max(out[6]))
                stdhr.append(np.std(out[6]))
            except:
                print("Ecg analysis got its problem for i={}", i)
                meanhr.append(np.nan)
                minhr.append(np.nan)
                maxhr.append(np.nan)
                stdhr.append(np.nan)
            calc_RR(out)
            IBI.append(measures['ibi'])
            pNN20.append(measures['pnn20'])
            pNN50.append(measures['pnn50'])
            SDNN.append(measures['sdnn'])
            SDSD.append(measures['sdsd'])
            RMSSD.append(measures['rmssd'])
        full_validation_features = pd.DataFrame()
        full_validation_features['heart_rate'] = pd.Series(meanhr)
        full_validation_features['minhr'] = pd.Series(minhr)
        full_validation_features['maxhr'] = pd.Series(maxhr)
        full_validation_features['stdhr'] = pd.Series(stdhr)
        full_validation_features['ibi'] = pd.Series(IBI)
        # full_validation_features['pnn20'] = pd.Series(pNN20)
        full_validation_features['pnn50'] = pd.Series(pNN50)
        full_validation_features['sdnn'] = pd.Series(SDNN)
        full_validation_features['sdsd'] = pd.Series(SDSD)
        full_validation_features['rmssd'] = pd.Series(RMSSD)

        full_validation_features = full_validation_features.to_numpy()
        full_validation_features, self.full_Y = self.data_factory.impute_dataset(full_validation_features, self.full_Y,
                                                                          impute_method=self.imputer)

        self.log_factory.InfoLog("After feature selection, the shape of X = {}".format(full_validation_features.shape))
        # full_validation_X, self.full_Y = self.data_factory.post_impute(full_validation_X, self.full_Y, impute_method=self.imputer)
        self.full_X = full_validation_features[:self.full_Y.shape[0], :]
        self.validation_X = full_validation_features[self.full_Y.shape[0]:, :]
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
                                        cv=5,
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

            seed = 77
            svmc = svm.SVC(class_weight='balanced',
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
            name = 'data/y_validation.csv'
            path = os.path.join('.', name)
            df.to_csv(path, index=False)

            # self.dump_validated_y(predicted_y_validate)
            # self.log_factory.InfoLog("all score = {}".format(f1_score(full_Y, predicted_y_full, average='micro')))
        elif self.model_name == 'ensemble':
            indicator = np.array([False for i in range(self.full_X.shape[0])])
            idx = [i for i in range(self.full_X.shape[0])]
            sampled_idx = random.sample(idx, self.full_Y.shape[0])
            indicator[sampled_idx[:int(self.train_percent * self.full_X.shape[0])]] = True
            train_X = self.full_X[indicator == True, :]
            train_Y = self.full_Y[indicator == True].reshape(-1)
            test_X = self.full_X[indicator == False, :]
            test_Y = self.full_Y[indicator == False].reshape(-1)
            self.log_factory.InfoLog("trainX: {}".format(train_X.shape))
            self.log_factory.InfoLog("testX: {}".format(test_X.shape))

            cls0, cls1, cls2, cls3 = np.bincount(self.full_Y.reshape(-1))
            self.log_factory.InfoLog("cls0: {}".format(cls0))
            self.log_factory.InfoLog("cls1: {}".format(cls1))
            self.log_factory.InfoLog("cls2: {}".format(cls2))
            self.log_factory.InfoLog("cls3: {}".format(cls3))

            cls0_bool = train_Y == 0
            cls1_bool = train_Y == 1
            cls2_bool = train_Y == 2
            cls3_bool = train_Y == 3

            train_X_y = np.concatenate((train_X, train_Y.reshape((-1, 1))), axis=1)

            cls0_X = train_X_y[cls0_bool.reshape(-1), :]  # 3030
            cls1_X = train_X_y[cls1_bool.reshape(-1), :]  # 443
            cls2_X = train_X_y[cls2_bool.reshape(-1), :]  # 1474
            cls3_X = train_X_y[cls3_bool.reshape(-1), :]  # 170

            # [0]. svm
            seed = 77
            params_svmc = {'gamma': [0.057, 0.08, 0.3, 0.6],
                           'degree': [3, 4, 5, 6, 7],
                           'kernel': ['poly', 'rbf'],
                           'class_weight': ['balanced', 'None'],
                           'C': [0.125,  0.6225, 0.875]}
            svmc = svm.SVC(random_state=seed,
                           tol=0.001,
                           decision_function_shape='ovo')
            svmc_gs = GridSearchCV(svmc, params_svmc, cv=5)
            svmc_gs.fit(train_X, train_Y)
            svmc_best = svmc_gs.best_estimator_
            self.log_factory.InfoLog("0. SVM: got its best parameter = {}".format(svmc_gs.best_params_))
            self.log_factory.InfoLog("0. SVM: got its score = {}".format(svmc_best.score(test_X, test_Y)))

            # [1]. Random Forest Classifier
            rf = RandomForestClassifier()
            params_rf = {'n_estimators': [50, 100, 200],
                         'class_weight': [None, 'balanced']}
            rf_gs = GridSearchCV(rf, params_rf, cv=5)
            rf_gs.fit(train_X, train_Y)
            rf_best = rf_gs.best_estimator_
            self.log_factory.InfoLog("1. Random Forest: got its best parameter = {}".format(rf_gs.best_params_))
            self.log_factory.InfoLog("1. Random Forest: got its score = {}".format(rf_best.score(test_X, test_Y)))

            # [2]. Logistic regression as classification
            lr = LogisticRegression(tol=0.001)
            params_lr = {'penalty': ['l1', 'l2', 'elasticnet'],
                         'class_weight': [None, 'balanced'],
                         'C': [0.375, 0.6225, 1.0]}
            lr_gs = GridSearchCV(lr, params_lr, cv=5)
            lr_gs.fit(train_X, train_Y)
            lr_best = lr_gs.best_estimator_
            self.log_factory.InfoLog("2. Logistic Regression: got its best parameter = {}".format(lr_gs.best_params_))
            self.log_factory.InfoLog("2. Logistic Regression: got its score = {}".format(lr_best.score(test_X, test_Y)))

            # [3]. KNN Classification
            knn = KNeighborsClassifier()
            params_knn = {'n_neighbors': [5, 10, 20],
                          'weights': ['uniform', 'distance'],
                          'p': [1, 2]}
            knn_gs = GridSearchCV(knn, params_knn, cv=5)
            knn_gs.fit(train_X, train_Y)
            knn_best = knn_gs.best_estimator_
            self.log_factory.InfoLog("3. KNN: got its best parameter = {}".format(knn_gs.best_params_))
            self.log_factory.InfoLog("3. KNN: got its score = {}".format(knn_best.score(test_X, test_Y)))

            # [4]. Ridge Classification
            rc = RidgeClassifier()
            params_rc = {'alpha': [0.625, 0.875, 1.0],
                         'class_weight': [None, 'balanced']}
            rc_gs = GridSearchCV(rc, params_rc, cv=5)
            rc_gs.fit(train_X, train_Y)
            rc_best = rc_gs.best_estimator_
            self.log_factory.InfoLog("4. Ridge Classifier: got its best parameter = {}".format(rc_gs.best_params_))
            self.log_factory.InfoLog("4. Ridge Classifier: got its score = {}".format(rc_best.score(test_X, test_Y)))

            # [final]. Voting Classification
            estimators = [('svmc', svmc_best), ('rf', rf_best), ('lr', lr_best)]
            ensemble = VotingClassifier(estimators, voting='hard')
            ensemble.fit(train_X, train_Y)
            self.log_factory.InfoLog("Finally, we got a score of {}".format(ensemble.score(test_X, test_Y)))
            self.log_factory.InfoLog(
                    "Finally, we got our f1 score on test set = {}".format(f1_score(test_Y, ensemble.predict(test_X), average='micro')))

            pred_val_y = ensemble.predict(self.validation_X)

            ID = np.array(range(len(self.validation_X)))
            df = pd.DataFrame({'id': ID,
                               'y': pred_val_y})
            name = 'y_validation.csv'
            path = os.path.join(self.data_path, name)
            df.to_csv(path, index=False)
        elif self.model_name == 'nnet':
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
                test_f1_score = 0.0
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
                        test_f1_score += f1_score(test_Y.cpu().numpy(), torch.floor(predicted_y_test).to(torch.int).cpu().numpy(), average='micro') / self.k_fold

                if epoch % 200 == 0:
                    self.log_factory.InfoLog(
                        "Epoch={}, while test loss={}, train loss={}, test MSE={}, train MSE={}, r2_score={}".format(
                            epoch, test_loss, train_loss, test_mse, train_mse, test_f1_score))
                    computed_losses.append(test_loss.detach().clone().cpu())
                    train_losses.append(train_loss)
                    with torch.no_grad():
                        predicted_y_validate = torch.floor(self.train_model(self.validation_X)).squeeze(1).to(torch.int).cpu().numpy()
                        self.dump_validated_y(predicted_y_validate)


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
