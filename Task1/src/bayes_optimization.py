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
from .data_utils.gain_imputer import GAINImputer

from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn import ensemble    
from sklearn.tree import DecisionTreeRegressor 
import xgboost as xgb
import lightgbm as lgb



class Bayes_Optimization:
    def __init__(self, core_management):
        self.core_management = core_management

        self.initialized = False


    #贝叶斯优化调节lasso参数
    def Bayes_opt_lasso(self, train_X, train_Y):
        #黑盒函数 
        def black_box_function(alpha):
            val = cross_val_score(
                make_pipeline(RobustScaler(), Lasso(alpha = alpha, random_state= 1)),
                train_X, train_Y,scoring='r2', cv=5, n_jobs=-1
            ).mean()
            return val  
        #定义域
        pbounds= {'alpha': (0, 1)}
                  #'bootstrap': [True, False]
        #实例化对象
        optimizer = BayesianOptimization(f= black_box_function,
                    pbounds= pbounds,
                    verbose= 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state= 1,
                    )
        #确定迭代次数
        optimizer.maximize(init_points= 5,  #执行随机搜索的步数
                           n_iter= 50,   #执行贝叶斯优化的步数
                           )
        #输出最优结果
        print(optimizer.max)
        alpha=optimizer.max['params']['alpha']            
        return alpha
  
    
    #贝叶斯优化调节Adaboost参数 45mins
    def Bayes_opt_Adaboost(self, train_X, train_Y):
        #黑盒函数 
        def black_box_function(n_estimators, learning_rate, min_samples_split,  max_features, max_depth, min_samples_leaf):
            val = cross_val_score(
                ensemble.AdaBoostRegressor(DecisionTreeRegressor(
                max_features = int(max_features),
                max_depth = int(max_depth),
                min_samples_split = int(min_samples_split),
                min_samples_leaf = int(min_samples_leaf),        
                random_state = 2),
                n_estimators = int(n_estimators),learning_rate = learning_rate),
                train_X, train_Y,scoring='r2', cv=5, n_jobs=-1
            ).mean()
            return val  
        #定义域
        pbounds= {'n_estimators': (50, 1500),
                  'learning_rate': (0.0000000001, 1),
                  'max_features': (1, train_X.shape[1]),
                  'max_depth': (2, 150),
                  'min_samples_split': (2, 30),
                  'min_samples_leaf':(1, 20)}
        #实例化对象
        optimizer = BayesianOptimization(f= black_box_function,
                    pbounds= pbounds,
                    verbose= 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state= 1,
                    )
        #确定迭代次数
        optimizer.maximize(init_points= 6,  #执行随机搜索的步数
                           n_iter= 60,   #执行贝叶斯优化的步数
                           )
        #输出最优结果
        print(optimizer.max)
        n_es = int(optimizer.max['params']['n_estimators'])
        l_ra = optimizer.max['params']['learning_rate']
        max_dep = int(optimizer.max['params']['max_depth'])
        max_fea = int(optimizer.max['params']['max_features'])
        min_s_l = int(optimizer.max['params']['min_samples_leaf'])
        min_s_s = int(optimizer.max['params']['min_samples_split'])
        return n_es, l_ra, max_dep, max_fea, min_s_l, min_s_s

    
    #贝叶斯优化调节RandomForest参数 17mins
    def Bayes_opt_RandomForest(self, train_X, train_Y):
        #黑盒函数 
        def black_box_function(n_estimators, min_samples_split,  max_features, max_depth, min_samples_leaf):
            val = cross_val_score(
                ensemble.RandomForestRegressor(
                max_features = int(max_features),
                max_depth = int(max_depth),
                min_samples_split = int(min_samples_split),
                min_samples_leaf = int(min_samples_leaf),        
                random_state = 2,
                n_estimators = int(n_estimators),
                oob_score = True),
                train_X, train_Y,scoring='r2', cv=5, n_jobs=-1
            ).mean()
            return val 
        #定义域
        pbounds= {'n_estimators': (50, 1500),
                  'max_features': (1, train_X.shape[1]),
                  'max_depth': (2, 150),
                  'min_samples_split': (2, 30),
                  'min_samples_leaf':(1, 20)}
        #实例化对象
        optimizer = BayesianOptimization(f= black_box_function,
                    pbounds= pbounds,
                    verbose= 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state= 1,
                    )
        #确定迭代次数
        optimizer.maximize(init_points= 6,  #执行随机搜索的步数
                           n_iter= 60,   #执行贝叶斯优化的步数
                           )
        #输出最优结果
        print(optimizer.max)
        n_es = int(optimizer.max['params']['n_estimators'])
        max_dep = int(optimizer.max['params']['max_depth'])
        max_fea = int(optimizer.max['params']['max_features'])
        min_s_l = int(optimizer.max['params']['min_samples_leaf'])
        min_s_s = int(optimizer.max['params']['min_samples_split'])
        return n_es, max_dep, max_fea, min_s_l, min_s_s

        
    #贝叶斯优化调节GBoost参数 30个 1h
    def Bayes_opt_GBoost(self, train_X, train_Y):
        #黑盒函数 
        def black_box_function(n_estimators, learning_rate, min_samples_split,  max_features, max_depth, min_samples_leaf):
            val = cross_val_score(
                ensemble.GradientBoostingRegressor(
                max_features = int(max_features),
                learning_rate = learning_rate,
                max_depth = int(max_depth),
                min_samples_split = int(min_samples_split),
                min_samples_leaf = int(min_samples_leaf),        
                random_state = 5,
                n_estimators = int(n_estimators),
                loss='huber'),
                train_X, train_Y,scoring='r2', cv=5, n_jobs=-1
            ).mean()
            return val  
        #定义域
        pbounds= {'n_estimators': (50, 3500),
                  'learning_rate': (0.0000000001, 1),
                  'max_features': (1, train_X.shape[1]),
                  'max_depth': (2, 150),
                  'min_samples_split': (2, 30),
                  'min_samples_leaf':(1, 20)}
        #实例化对象
        optimizer = BayesianOptimization(f= black_box_function,
                    pbounds= pbounds,
                    verbose= 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state= 1,
                    )
        #确定迭代次数
        optimizer.maximize(init_points= 5,  #执行随机搜索的步数 
                           n_iter= 30,   #执行贝叶斯优化的步数  
                           )
        #输出最优结果
        print(optimizer.max)
        n_es = int(optimizer.max['params']['n_estimators'])
        l_ra = optimizer.max['params']['learning_rate']
        max_dep = int(optimizer.max['params']['max_depth'])
        max_fea = int(optimizer.max['params']['max_features'])
        min_s_l = int(optimizer.max['params']['min_samples_leaf'])
        min_s_s = int(optimizer.max['params']['min_samples_split'])
        return n_es, l_ra, max_dep, max_fea, min_s_l, min_s_s
    
    
    #贝叶斯优化调节xgb参数
    def Bayes_opt_xgb(self, train_X, train_Y):
        #黑盒函数 
        def black_box_function(n_estimators, learning_rate, max_depth, min_child_weight, gamma, 
                               subsample, colsample_bytree, reg_alpha, reg_lambda):
            val = cross_val_score(
                xgb.XGBRegressor(n_estimators = int(n_estimators),
                learning_rate = learning_rate,
                max_depth = int(max_depth),
                min_child_weight = min_child_weight,
                gamma = gamma,
                subsample = subsample,
                colsample_bytree = colsample_bytree,
                reg_alpha = reg_alpha,
                reg_lambda = reg_lambda,
                random_state = 7,
                silent = 1),
                train_X, train_Y,scoring='r2', cv=5, n_jobs=-1
            ).mean()
            return val 
        #定义域
        pbounds= {'n_estimators': (50, 3500),
                  'learning_rate': (0.0000000001, 1),
                  'max_depth': (2, 150),
                  'min_child_weight':  (1,50),
                  'gamma': (0,1),
                  'subsample': (0,1),
                  'colsample_bytree':(0,1),
                  'reg_alpha': (0,1),
                  'reg_lambda': (0,1)}
        #实例化对象
        optimizer = BayesianOptimization(f= black_box_function,
                    pbounds= pbounds,
                    verbose= 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state= 1,
                    )
        #确定迭代次数
        optimizer.maximize(init_points= 2,  #执行随机搜索的步数 6
                           n_iter= 4,   #执行贝叶斯优化的步数 60
                           )
        #输出最优结果
        print(optimizer.max)
        n_es = int(optimizer.max['params']['n_estimators'])
        l_ra = optimizer.max['params']['learning_rate']
        max_dep = int(optimizer.max['params']['max_depth'])
        min_c_w = optimizer.max['params']['min_child_weight']
        gam = optimizer.max['params']['gamma']
        subs = optimizer.max['params']['subsample']
        col_b = optimizer.max['params']['colsample_bytree']
        reg_a = optimizer.max['params']['reg_alpha']
        reg_l = optimizer.max['params']['reg_lambda']
        return n_es, l_ra, max_dep, min_c_w, gam, subs, col_b, reg_a, reg_l
 