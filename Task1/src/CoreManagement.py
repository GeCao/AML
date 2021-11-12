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
from .bayes_optimization import Bayes_Optimization
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from bayes_opt import BayesianOptimization

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
        self.bayes_optimization = Bayes_Optimization(self)

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
            #from sklearn.neural_network import MLPRegressor

            row_idx = [i for i in range(self.full_X.shape[0])]
            random.shuffle(row_idx)
            train_X = self.full_X[0:math.floor(len(row_idx) * self.train_percent), ...].cpu().numpy()
            val_X = self.full_X[train_X.shape[0]:, ...].cpu().numpy()
            train_Y = self.full_Y[0:train_X.shape[0], ...].cpu().numpy()
            val_Y = self.full_Y[train_X.shape[0]:, ...].cpu().numpy()

            ###贝叶斯调参            
            #黑盒函数 
            def black_box_function(n_estimators, min_samples_split,  max_features, max_depth, min_samples_leaf):
                val = cross_val_score(
                    ExtraTreesRegressor(n_estimators = int(n_estimators),
                        max_features = int(max_features),
                        max_depth = int(max_depth),
                        min_samples_split = int(min_samples_split),
                        min_samples_leaf = int(min_samples_leaf),
                        random_state = 2,
                        bootstrap=True
                    ),
                    train_X, train_Y,scoring='r2', cv=5, n_jobs=-1
                ).mean()
                return val  #max_features = max_features, # float
            
            #定义域
            pbounds= {'n_estimators': (500, 2000),
                      'max_features': (1, self.full_X.shape[1]),
                      'max_depth': (5, 150),
                      'min_samples_split': (2, 30),
                      'min_samples_leaf':(1, 20)}
                      #'bootstrap': [True, False]
            #实例化对象
            optimizer = BayesianOptimization(f= black_box_function,
                        pbounds= pbounds,
                        verbose= 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                        random_state= 1,
                        )
            #确定迭代次数
            optimizer.maximize(init_points= 12,  #执行随机搜索的步数
                               n_iter= 100,   #执行贝叶斯优化的步数
                               )
            #输出最优结果
            print(optimizer.max)
            n_es=optimizer.max['params']['n_estimators']
            max_dep=optimizer.max['params']['max_depth']
            max_fea=optimizer.max['params']['max_features']
            min_s_l=optimizer.max['params']['min_samples_leaf']
            min_s_s=optimizer.max['params']['min_samples_split']
            
            # extra trees regression
            extra_tree = ExtraTreesRegressor(n_estimators=int(n_es),max_depth=int(max_dep), max_features=int(max_fea),
                 min_samples_leaf=int(min_s_l), min_samples_split=int(min_s_s), n_jobs=-1,bootstrap=True)
            extra_tree.fit(train_X, train_Y)
            extra_pred = extra_tree.predict(val_X)

            self.log_factory.InfoLog("The score of extra_tree for validation={}".format(r2_score(val_Y, extra_pred)))
            
            #导出正确格式的csv文件
            ID = np.array(range(len(val_X)))
            import pandas as pd
            df = pd.DataFrame({'id': ID,
                               'y': extra_pred})
            df.to_csv(os.path.join(self.data_path, 'prediction.csv'), index=False)
            self.dump_validated_y(extra_tree.predict(self.validation_X.cpu().numpy()))
        
        elif self.model_name == "adaboost":
            from sklearn import ensemble 
            from sklearn.tree import DecisionTreeRegressor 
            row_idx = [i for i in range(self.full_X.shape[0])]
            random.shuffle(row_idx)
            train_X = self.full_X[0:math.floor(len(row_idx) * self.train_percent), ...].cpu().numpy()
            val_X = self.full_X[train_X.shape[0]:, ...].cpu().numpy()
            train_Y = self.full_Y[0:train_X.shape[0], ...].cpu().numpy()
            val_Y = self.full_Y[train_X.shape[0]:, ...].cpu().numpy()
            
            print('Bayes_Optimization(adaboost)')
            n_es, l_ra, max_dep, max_fea, min_s_l, min_s_s=self.bayes_optimization.Bayes_opt_Adaboost(train_X = train_X, train_Y = train_Y)            
            Adaboost = ensemble.AdaBoostRegressor(
                DecisionTreeRegressor( max_features = max_fea, max_depth = max_dep, 
                min_samples_split = min_s_s,min_samples_leaf = min_s_l, random_state = 2),
                n_estimators = n_es,learning_rate = l_ra)
                
            Adaboost.fit(train_X, train_Y)
            ada_pred = Adaboost.predict(val_X)

            self.log_factory.InfoLog("The score of Adaboost for validation={}".format(r2_score(val_Y, ada_pred)))
            
            #导出正确格式的csv文件
            ID = np.array(range(len(val_X)))
            import pandas as pd
            df = pd.DataFrame({'id': ID,
                               'y': ada_pred})
            df.to_csv(os.path.join(self.data_path, 'prediction.csv'), index=False)
            self.dump_validated_y(Adaboost.predict(self.validation_X.cpu().numpy()))
            
        elif self.model_name == "Gboost":
            from sklearn import ensemble    
            model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor()
            
            row_idx = [i for i in range(self.full_X.shape[0])]
            random.shuffle(row_idx)
            train_X = self.full_X[0:math.floor(len(row_idx) * self.train_percent), ...].cpu().numpy()
            val_X = self.full_X[train_X.shape[0]:, ...].cpu().numpy()
            train_Y = self.full_Y[0:train_X.shape[0], ...].cpu().numpy()
            val_Y = self.full_Y[train_X.shape[0]:, ...].cpu().numpy()
            
            #Gboost
            print('Bayes_Optimization(Gboost)')
            n_es, l_ra, max_dep, max_fea, min_s_l, min_s_s = self.bayes_optimization.Bayes_opt_GBoost(train_X = train_X, train_Y = train_Y)            
            Gboost = ensemble.GradientBoostingRegressor(max_features = max_fea, max_depth = max_dep, 
                min_samples_split = min_s_s, min_samples_leaf = min_s_l, random_state = 2,
                n_estimators = n_es, learning_rate = l_ra, loss='huber')

            Gboost.fit(train_X, train_Y)
            gbt_pred = Gboost.predict(val_X)
            
            self.log_factory.InfoLog("The score of Adaboost for validation={}".format(r2_score(val_Y, gbt_pred)))
            
            #导出正确格式的csv文件
            ID = np.array(range(len(val_X)))
            import pandas as pd
            df = pd.DataFrame({'id': ID,
                               'y': gbt_pred})
            df.to_csv(os.path.join(self.data_path, 'prediction.csv'), index=False)
            self.dump_validated_y(Gboost.predict(self.validation_X.cpu().numpy()))
            
        elif self.model_name == "ensemble":
            from sklearn.model_selection import KFold, GridSearchCV
            
            row_idx = [i for i in range(self.full_X.shape[0])]
            random.shuffle(row_idx)
            train_X = self.full_X[0:math.floor(len(row_idx) * self.train_percent), ...].cpu().numpy()
            val_X = self.full_X[train_X.shape[0]:, ...].cpu().numpy()
            train_Y = self.full_Y[0:train_X.shape[0], ...].cpu().numpy()
            val_Y = self.full_Y[train_X.shape[0]:, ...].cpu().numpy()
            
            score_function = r2_score

            # =============Add different models here!!!!=============
            model_heads = []
            models = []
            from sklearn import tree    # 0
            model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
            model_heads.append("Decision Tree Regression\t\t")
            models.append(model_DecisionTreeRegressor)
            
            from sklearn import linear_model    # 1
            model_LinearRegression = linear_model.LinearRegression()
            model_heads.append("Linear Regression\t\t\t\t")
            models.append(model_LinearRegression)
            
            from sklearn import svm     # 2
            model_SVR = svm.SVR()
            model_heads.append("Support Vector Machine Regression")
            models.append(model_SVR)
            
            from sklearn import neighbors   # 3
            model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
            model_heads.append("K-Nearest Neighbor Regression\t")
            models.append(model_KNeighborsRegressor)
            
            from sklearn import ensemble    # 4
            model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
            model_heads.append("Random Forest Regression\t\t")
            models.append(model_RandomForestRegressor)
            
            from sklearn import ensemble    # 5
            model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=150)
            model_heads.append("AdaBoost Regression\t\t\t\t")
            models.append(model_AdaBoostRegressor)
            
            from sklearn import ensemble    # 6
            model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor()
            model_heads.append("Gradient Boosting Regression\t")
            models.append(model_GradientBoostingRegressor)
            
            from sklearn.ensemble import BaggingRegressor   # 7
            model_BaggingRegressor = BaggingRegressor()
            model_heads.append("Bagging Regression\t\t\t\t")
            models.append(model_BaggingRegressor)
            
            from sklearn.tree import ExtraTreeRegressor     # 8
            model_ExtraTreeRegressor = ExtraTreeRegressor()
            model_heads.append("ExtraTree Regression\t\t\t")
            models.append(model_ExtraTreeRegressor)
            
            import xgboost as xgb       # 9
            # params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 8, 'min_child_weight': 2, 'seed': 0,
            #           'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 2}
            model_XGBoostRegressor = xgb.XGBRegressor()
            model_heads.append("XGBoost Regression\t\t\t\t")
            models.append(model_XGBoostRegressor)
            # =============Model Adding Ends=============
            
            # =============For Esemble and Stacking =============
            from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
            from sklearn.kernel_ridge import KernelRidge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import RobustScaler
            from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
            from sklearn.model_selection import KFold
            import xgboost as xgb
            import lightgbm as lgb
            from sklearn import linear_model
            from sklearn.tree import DecisionTreeRegressor 
            from sklearn.ensemble import ExtraTreesRegressor


            
            #原组合：Enet+KRR+GBoost+lasso(meta)+xgb+lgb
            #新组合：adaboost+RandomForest+GBoost+lasso(meta)+xgb+lgb
            '''
            #lasso
            print('Bayes_Optimization(lasso)')
            alp = self.bayes_optimization.Bayes_opt_lasso(train_X = train_X, train_Y = train_Y)
            lasso = make_pipeline(RobustScaler(), Lasso(alpha = alp, random_state=1))
            #lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
            '''
            #extra_tree
            print('Bayes_Optimization(extra_tree)')
            n_es, max_dep, max_fea, min_s_l, min_s_s=self.bayes_optimization.Bayes_opt_extratree(train_X = train_X, train_Y = train_Y) 
            extra_tree = ExtraTreesRegressor(n_estimators=int(n_es),max_depth=int(max_dep), max_features=int(max_fea),
             min_samples_leaf=int(min_s_l), min_samples_split=int(min_s_s), n_jobs=-1,bootstrap=True)
            #adaboost
            print('Bayes_Optimization(adaboost)')
            n_es, l_ra, max_dep, max_fea, min_s_l, min_s_s=self.bayes_optimization.Bayes_opt_Adaboost(train_X = train_X, train_Y = train_Y)            
            Adaboost = ensemble.AdaBoostRegressor(
                DecisionTreeRegressor( max_features = max_fea, max_depth = max_dep, 
                min_samples_split = min_s_s,min_samples_leaf = min_s_l, random_state = 2),
                n_estimators = n_es,learning_rate = l_ra)
            #RandomForest
            print('Bayes_Optimization(RandomForest)')
            n_es, max_dep, max_fea, min_s_l, min_s_s=self.bayes_optimization.Bayes_opt_RandomForest(train_X = train_X, train_Y = train_Y) 
            RandomForest = ensemble.RandomForestRegressor(n_estimators = n_es,
              max_features = max_fea, max_depth = max_dep, 
              min_samples_split = min_s_s, min_samples_leaf = min_s_l, 
              random_state = 2)
            #Gboost
            print('Bayes_Optimization(Gboost)')
            n_es, l_ra, max_dep, max_fea, min_s_l, min_s_s = self.bayes_optimization.Bayes_opt_GBoost(train_X = train_X, train_Y = train_Y)            
            Gboost = ensemble.GradientBoostingRegressor(max_features = max_fea, max_depth = max_dep, 
                min_samples_split = min_s_s, min_samples_leaf = min_s_l, random_state = 2,
                n_estimators = n_es, learning_rate = l_ra, loss='huber')
            #xgb
            model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                         learning_rate=0.05, max_depth=3,
                                         min_child_weight=1.7817, n_estimators=2200,
                                         reg_alpha=0.4640, reg_lambda=0.8571,
                                         subsample=0.5213, silent=1,
                                         random_state =7, nthread = -1)
            #lgb
            model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                          learning_rate=0.05, n_estimators=720,
                                          max_bin = 55, bagging_fraction = 0.8,
                                          bagging_freq = 5, feature_fraction = 0.2319,
                                          feature_fraction_seed=9, bagging_seed=9,
                                          min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                        
            
            def get_model_score(model, x_all, y_all, n_folds=5):
                #交叉验证求r2_score
                score_func = r2_score
                kf = KFold(n_splits=n_folds, shuffle=True)
                score_mean_test = 0
                score_mean_train = 0
                for train_idx, test_idx in kf.split(x_all):
                    x_train = x_all[train_idx]
                    y_train = y_all[train_idx]
                    x_test = x_all[test_idx]
                    y_test = y_all[test_idx]
                    score_test, score_train = try_different_method(model, x_train, y_train, x_test, y_test, score_func)
                    score_mean_test += score_test
                    score_mean_train += score_train
                score_mean_test /= n_folds
                score_mean_train /= n_folds
                return score_mean_test
            
            
            def try_different_method(model, x_train, y_train, x_test, y_test, score_func):
                #求模型分数
                """
                Inner function in train_evaluate_return_best_model for model training.
                :param model: one specific model
                :param x_train:
                :param y_train:
                :param x_test:
                :param y_test:
                :param score_func:
                :return score:
                """
                model.fit(x_train, y_train)
                result_test = model.predict(x_test)
                result_train = model.predict(x_train)
                return score_func(y_test, result_test), score_func(y_train, result_train)
    
    
            class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
                #定义StackingAveragedModels
                """
                from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
                """
                def __init__(self, base_models, meta_model, n_folds=5):
                    self.base_models = base_models
                    self.meta_model = meta_model
                    self.n_folds = n_folds
            
                # We again fit the data on clones of the original models
                def fit(self, X, y):
                    self.base_models_ = [list() for x in self.base_models]
                    self.meta_model_ = clone(self.meta_model)
                    kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
            
                    # Train cloned base models then create out-of-fold predictions
                    # that are needed to train the cloned meta-model
                    out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
                    for i, model in enumerate(self.base_models):
                        for train_index, holdout_index in kfold.split(X, y):
                            instance = clone(model)
                            self.base_models_[i].append(instance)
                            instance.fit(X[train_index], y[train_index])
                            y_pred = instance.predict(X[holdout_index])
                            out_of_fold_predictions[holdout_index, i] = y_pred.ravel()
            
                    # Now train the cloned  meta-model using the out-of-fold predictions as new feature
                    self.meta_model_.fit(out_of_fold_predictions, y)
                    return self
            
                # Do the predictions of all base models on the test data and use the averaged predictions as
                # meta-features for the final prediction which is done by the meta-model
                def predict(self, X):
                    meta_features = np.column_stack([
                        np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                        for base_models in self.base_models_])
                    return self.meta_model_.predict(meta_features)
                
            # =============For Esemble and Stacking(end)=============
            
            
             
            def train_evaluate_return_best_model(x_all, y_all, score_func=r2_score, fold_num=5, return_ave=False):
                """
                Train predefined models on data using 5-fold validation
                :param x_all: ndarray containing all features
                :param y_all: ndarray containing all labels
                :param score_func: score function
                :param fold_num: fold number to use K-fold CV
                :param return_ave: return average performance on all methods?
                :return best_model: best model trained on all data
                """
                print()
                print("Training model with K-fords...")
                kf = KFold(n_splits=fold_num, shuffle=True)
                best_score = 0
                best_idx = 0
                ave_score = 0
                for (model_idx, model) in enumerate(models):
                    score_mean_test = 0
                    score_mean_train = 0
                    for train_idx, test_idx in kf.split(x_all):
                        x_train = x_all[train_idx]
                        y_train = y_all[train_idx]
                        x_test = x_all[test_idx]
                        y_test = y_all[test_idx]
                        score_test, score_train = try_different_method(model, x_train, y_train, x_test, y_test, score_func)
                        score_mean_test+=score_test
                        score_mean_train+=score_train
                    score_mean_test /= fold_num
                    score_mean_train /= fold_num
                    ave_score += score_mean_test
                    if not return_ave:
                        print("{} \t score train: {}, score test: {}".format(model_heads[model_idx], score_mean_train, score_mean_test))
                    if best_score < score_mean_test:
                        best_score = score_mean_test
                        best_idx = model_idx
                print("Training done")
                print("Best model: {}\t Score: {}".format(model_heads[best_idx], best_score))
                if return_ave:
                    print("Average score on {} models = {}".format(len(models), ave_score/len(models)))
                best_model = models[best_idx]
                best_model.fit(x_all, y_all)
                return best_idx, best_model
            
            def tune_model_params(x_all, y_all):
                """
                Tune models on data using 5-fold validation
                :param x_all: ndarray containing all features
                :param y_all: ndarray containing all labels
                :param score_func: score function
                :param fold_num: fold number to use K-fold CV
                :return best_model: best model trained on all data
                """
                print()
                print("Tuning model...")
                cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
                other_params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 8, 'min_child_weight': 2, 'seed': 0,
                                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 2}
                model = xgb.XGBRegressor(**other_params)
                optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=1)
                optimized_GBM.fit(x_all, y_all)
                evalute_result = optimized_GBM.grid_scores_
                print('Result:{0}'.format(evalute_result))
                print('Best params：{0}'.format(optimized_GBM.best_params_))
                print('Best score:{0}'.format(optimized_GBM.best_score_))
                
            def get_model(x_all, y_all, model_idx):
                """
                Given model index return the corresponding model trained on all data
                :param x_all:
                :param y_all:
                :param model_idx:
                :return model:
                """
                print()
                print("Training with all data using {}".format(model_heads[model_idx]))
                model = models[model_idx].fit(x_all, y_all)
                return model
            
            
            '''
            print('Find best models:')
            find_best_model = False     # display several preselected models' results (5-folds)
            if find_best_model:
                # show some results
                _, _ = train_evaluate_return_best_model(x_all=train_X, y_all=train_Y,
                                                        score_func=score_function, fold_num=5)
            '''
            # =================================================
            # Ensemble + stacking
            # =================================================
            print()
            print("Ensemble start...")
            '''
            score = get_model_score(lasso, train_X, train_Y)
            print("\nLasso score: {:.4f}\n".format(score))
            '''
            score = get_model_score(extra_tree, train_X, train_Y)
            print("\nextra_tree score: {:.4f}\n".format(score))
            score = get_model_score(Adaboost, train_X, train_Y)
            print("Adaboost score: {:.4f}\n".format(score))
            score = get_model_score(RandomForest, train_X, train_Y)
            print("Randomforest score: {:.4f}\n".format(score))
            score = get_model_score(Gboost, train_X, train_Y)
            print("Gradient Boosting score: {:.4f}\n".format(score))
            score = get_model_score(model_xgb, train_X, train_Y)
            print("Xgboost score: {:.4f}\n".format(score))
            score = get_model_score(model_lgb, train_X, train_Y)
            print("LGBM score: {:.4f}\n".format(score))
                
            
            #stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                             #meta_model=lasso)
            stacked_averaged_models = StackingAveragedModels(base_models=(Adaboost, RandomForest, Gboost),
                                                             meta_model=extra_tree)                                                 
            score = get_model_score(stacked_averaged_models, train_X, train_Y)
            print("Stacking Averaged models score: {:.4f}".format(score))
            stacked_averaged_models.fit(train_X, train_Y)
            stacked_train_pred = stacked_averaged_models.predict(train_X)
            stacked_pred = stacked_averaged_models.predict(val_X)
            print('r2 score of stack models on train data:', r2_score(train_Y, stacked_train_pred))
            model_xgb.fit(train_X, train_Y)
            xgb_train_pred = model_xgb.predict(train_X)
            xgb_pred = model_xgb.predict(val_X)
            print('r2 score of xgb on train data:', r2_score(train_Y, xgb_train_pred))
            model_lgb.fit(train_X, train_Y)
            lgb_train_pred = model_lgb.predict(train_X)
            lgb_pred = model_lgb.predict(val_X)
            print('r2 score of lgb on train data:', r2_score(train_Y, lgb_train_pred))
            print('r2 score on train data:')
            print(r2_score(train_Y, stacked_train_pred * 0.70 +
                           xgb_train_pred * 0.15 + lgb_train_pred * 0.15))
            model_ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
            
            self.log_factory.InfoLog("The score of ensemble for validation={}".format(r2_score(val_Y, model_ensemble)))
            #导出正确格式的csv文件
            ID = np.array(range(len(val_X)))
            import pandas as pd
            df = pd.DataFrame({'id': ID,
                               'y': model_ensemble})
            df.to_csv(os.path.join(self.data_path, 'prediction.csv'), index=False)
            self.dump_validated_y(
                stacked_averaged_models.predict(self.validation_X.cpu().numpy()) * 0.70 
                + model_xgb.predict(self.validation_X.cpu().numpy()) * 0.15 
                + model_lgb.predict(self.validation_X.cpu().numpy()) * 0.15)
            #==============ensemble模型结束======================
            
            
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
            

    

