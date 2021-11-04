import os
import numpy as np
import torch

from .data_factory import DataFactory
from .log_factory import LogFactory
from .Normalizer import UnitGaussianNormalizer
from .models.lasso_model import MyLasso
from .utils import *
from sklearn.metrics import r2_score


class CoreComponent:
    def __init__(self, model='lasso', imputer=None, outlier=None, pca=None, device=None):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.imputer = 'knn' if imputer is None else imputer
        self.outlier = 'zscore' if outlier is None else outlier
        self.pca = 'pca' if outlier is None else pca
        self.device = 'cuda' if device is None else device  # choose with your preference

        model_name = 'lasso' if model is None else model  # choose with your preference
        if model_name == 'lasso':
            self.train_model = MyLasso(self)
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

        # 2. process X files together, while y files can not be processed since it is physically correct
        full_X_shape_0 = self.full_X.shape[0]
        validation_X_shape_0 = self.validation_X.shape[0]
        full_validation_X = np.concatenate((self.full_X, self.validation_X), axis=0)
        full_validation_X = self.data_factory.process_dataset(full_validation_X, impute_method=self.imputer,
                                                              outlier_method=self.outlier, pca_method=self.pca)
        self.full_normalizer.initialization(full_validation_X)
        full_validation_X = self.full_normalizer.encode(full_validation_X)
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

        self.train_model.initialization()
        self.initialized = True

    def run(self):
        computed_losses = []
        for epoch in range(self.train_model.total_epoch):
            stride = self.full_X.shape[0] // self.k_fold
            train_X, train_Y, test_X, test_Y = None, None, None, None

            test_loss = 0.0
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
                    predicted_y_test = self.train_model(test_X)
                    test_loss += self.train_model.compute_loss(predicted_y_test, test_Y)
                    test_r2_score = r2_score(test_Y.cpu().numpy(), predicted_y_test.cpu().numpy())

            if epoch % 200 == 0:
                self.log_factory.Slog(MessageAttribute.EInfo,
                                      sentences="Epoch={}, while loss={}, r2_score = {}".format(epoch, test_loss, test_r2_score))
                computed_losses.append(test_loss.detach().clone().cpu())
                with torch.no_grad():
                    predicted_y_validate = self.train_model(self.validation_X).squeeze(1).cpu().numpy()
                    # predicted_y_validate = self.y_normalizer.decode(predicted_y_validate)
                    self.log_factory.Slog(MessageAttribute.EInfo, sentences="Shape of predicted y={}".format(predicted_y_validate.shape))
                    with open(os.path.join(self.data_path, "y_validate.csv"), 'w') as f:
                        f.write("id,y\n")
                        for i, pred_y in enumerate(predicted_y_validate):
                            f.write("{},{}\n".format(i, pred_y))
                        f.close()
        model_evaluation(computed_losses, epoch_step=200)

    def kill(self):
        self.log_factory.kill()