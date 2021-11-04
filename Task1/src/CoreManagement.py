import os
import numpy as np
import torch

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
        else:
            self.train_model = None

        self.log_factory = LogFactory(self, log_to_disk=False)
        self.data_factory = DataFactory(self)
        self.full_normalizer = UnitGaussianNormalizer(self)
        self.y_normalizer = UnitGaussianNormalizer(self)

        self.full_X = None
        self.full_Y = None
        self.validation_X = None
        self.validation_Y = None

        self.k_fold = 5

        self.initialized = False

    def initialization(self):
        self.log_factory.initialization()
        self.log_factory.Slog(MessageAttribute.EInfo, sentences="Log Factory fully created")

        self.data_factory.initialization()

        # 1. read data
        self.full_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_train.csv"))
        self.full_Y = self.data_factory.read_dataset(os.path.join(self.data_path, "y_train.csv"))
        self.validation_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_test.csv"))

        # 2. process X files together
        full_X_shape_0 = self.full_X.shape[0]
        validation_X_shape_0 = self.validation_X.shape[0]
        full_validation_X = np.concatenate((self.full_X, self.validation_X), axis=0)
        full_validation_X = self.data_factory.process_dataset(full_validation_X, impute_method=self.imputer,
                                                              outlier_method=self.outlier, rows_X=full_X_shape_0)
        self.full_Y = self.data_factory.process_dataset(self.full_Y, impute_method=self.imputer,
                                                        outlier_method='else')
        self.full_normalizer.initialization(full_validation_X)
        full_validation_X = self.full_normalizer.encode(full_validation_X)
        full_X_shape_0 = len(self.full_Y)
        full_validation_X, self.full_Y = self.data_factory.feature_selection(full_validation_X, self.full_Y,
                                                                             method=self.pca, rows_X=full_X_shape_0)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="After feature selection, the shape of X = {}".format(full_validation_X.shape))
        self.full_X = full_validation_X[:full_X_shape_0, :]
        self.validation_X = full_validation_X[-validation_X_shape_0:, :]

        # self.y_normalizer.initialization(self.full_Y)
        # self.full_Y = self.y_normalizer.encode(self.full_Y)

        # 3. transfer numpy data to Tensor data
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from X_train.csv, with shape as {}".format(self.full_X.shape))
        self.full_X = torch.autograd.Variable(torch.from_numpy(np.array(self.full_X)).float()).to(self.device)
        # self.full_Y = self.data_factory.process_dataset(self.full_Y) # Y data cannot be processed!
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from y_train.csv, with shape as {}".format(self.full_Y.shape))
        self.full_Y = torch.autograd.Variable(torch.from_numpy(np.array(self.full_Y)).float()).to(self.device)

        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from X_test.csv, with shape as {}".format(self.validation_X.shape))
        self.validation_X = torch.autograd.Variable(torch.from_numpy(np.array(self.validation_X)).float()).to(self.device)

        self.initialized = True

    def run(self):
        if self.model_name == "lasso":
            full_X = self.full_X.cpu().numpy()
            full_Y = self.full_Y.cpu().squeeze(1).numpy()
            reg = self.train_model(n_alphas=10, cv=self.k_fold, eps=1e-3, max_iter=500).fit(full_X, full_Y)
            predicted_y_validate = reg.predict(self.validation_X.cpu().numpy())
            predicted_y_full = reg.predict(full_X)
            self.dump_validated_y(predicted_y_validate)
            self.log_factory.Slog(MessageAttribute.EInfo,
                                  "all score = {}".format(r2_score(full_Y, predicted_y_full)))
        elif self.model_name == "nnet":
            self.train_model.initialization()
            computed_losses = []
            train_losses = []
            for epoch in range(self.train_model.total_epoch):
                stride = self.full_X.shape[0] // self.k_fold
                train_X, train_Y, test_X, test_Y = None, None, None, None

                test_loss = 0.0
                train_loss = 0.0
                test_r2_score = 0.0
                self.train_model.train()
                for i in range(self.k_fold):
                    if i != self.k_fold - 1:
                        # k-fold CV
                        train_X = self.full_X[
                            (torch.arange(self.full_X.shape[0]) < i*stride) + (torch.arange(self.full_X.shape[0]) >= (i + 1)*stride)]
                        train_Y = self.full_Y[
                            (torch.arange(self.full_X.shape[0]) < i*stride) + (torch.arange(self.full_X.shape[0]) >= (i + 1)*stride)]
                        test_X = self.full_X[i*stride: (i + 1)*stride, :]
                        test_Y = self.full_Y[i*stride: (i + 1)*stride, :]
                    else:
                        train_X = self.full_X[:i*stride, ...]
                        train_Y = self.full_Y[:i*stride, ...]
                        test_X = self.full_X[i * stride:, ...]
                        test_Y = self.full_Y[i * stride:, ...]

                    self.train_model.optimizer.zero_grad()
                    predicted_y = self.train_model(train_X)
                    temp_loss = self.train_model.compute_loss(predicted_y, train_Y)
                    temp_loss.backward()
                    self.train_model.optimizer.step()

                    with torch.no_grad():
                        train_loss = temp_loss.detach().clone().cpu()
                        predicted_y_test = self.train_model(test_X)
                        test_loss += self.train_model.compute_loss(predicted_y_test, test_Y)
                        test_r2_score = r2_score(test_Y.cpu().numpy(), predicted_y_test.cpu().numpy())

                if epoch % 200 == 0:
                    self.log_factory.Slog(MessageAttribute.EInfo,
                            sentences="Epoch={}, while test loss={}, train loss = {}, r2_score = {}".format(epoch, test_loss, train_loss, test_r2_score))
                    computed_losses.append(test_loss.detach().clone().cpu())
                    train_losses.append(train_loss)
                    with torch.no_grad():
                        predicted_y_validate = self.train_model(self.validation_X).squeeze(1).cpu().numpy()
                        # predicted_y_validate = self.y_normalizer.decode(predicted_y_validate)
                        self.dump_validated_y(predicted_y_validate)
            model_evaluation(computed_losses, train_losses, epoch_step=200)

    def kill(self):
        self.log_factory.kill()

    def dump_validated_y(self, predicted_y_validate):
        try:
            predicted_y_validate = predicted_y_validate.cpu().numpy()
        except:
            pass

        with open(os.path.join(self.data_path, "y_validate.csv"), 'w') as f:
            f.write("id,y\n")
            for i, pred_y in enumerate(predicted_y_validate):
                f.write("{},{}\n".format(i, pred_y))
            f.close()