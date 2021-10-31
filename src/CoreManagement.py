import os
import numpy as np
import torch

from .data_factory import DataFactory
from .log_factory import LogFactory
from .Normalizer import UnitGaussianNormalizer
from .models.lasso_model import MyLasso
from .utils import *


class CoreComponent:
    def __init__(self, model='lasso', device=None):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.device = 'cuda' if device is None else device  # choose with your preference

        model_name = 'lasso' if model is None else model  # choose with your preference
        if model_name == 'lasso':
            self.train_model = MyLasso(self)
        else:
            self.train_model = None

        self.log_factory = LogFactory(self, log_to_disk=False)
        self.data_factory = DataFactory(self)
        self.full_normalizer = UnitGaussianNormalizer(self)
        self.validation_normalizer = UnitGaussianNormalizer(self)

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

        self.full_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_train.csv"))
        self.full_X = self.data_factory.process_dataset(self.full_X)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from X_train.csv, with shape as {}".format(self.full_X.shape))
        self.full_X = torch.autograd.Variable(torch.from_numpy(np.array(self.full_X)).float()).to(self.device)

        self.full_Y = self.data_factory.read_dataset(os.path.join(self.data_path, "y_train.csv"))
        self.full_Y = self.data_factory.process_dataset(self.full_Y)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from y_train.csv, with shape as {}".format(self.full_Y.shape))
        self.full_Y = torch.autograd.Variable(torch.from_numpy(np.array(self.full_Y)).float()).to(self.device)

        self.validation_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_test.csv"))
        self.validation_X = self.data_factory.process_dataset(self.validation_X)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from X_test.csv, with shape as {}".format(self.validation_X.shape))
        self.validation_X = torch.autograd.Variable(torch.from_numpy(np.array(self.validation_X)).float()).to(self.device)

        self.full_normalizer.initialization(self.full_X)
        self.full_X = self.full_normalizer.encode(self.full_X)

        self.validation_normalizer.initialization(self.validation_X)
        self.validation_X = self.validation_normalizer.encode(self.validation_X)

        self.train_model.initialization()

        self.initialized = True

    def run(self):
        computed_losses = []
        for epoch in range(self.train_model.total_epoch):
            stride = self.full_X.shape[0] // self.k_fold
            train_X = None
            train_Y = None
            test_X = None
            test_Y = None

            test_loss = 0.0
            for i in range(self.k_fold):
                if i != self.k_fold - 1:
                    train_X = self.full_X[
                        (torch.arange(self.full_X.shape[0]) < i*stride) + (torch.arange(self.full_X.shape[0]) >= (i + 1)*stride)]
                    train_Y = self.full_Y[
                        (torch.arange(self.full_X.shape[0]) < i*stride) + (torch.arange(self.full_X.shape[0]) >= (i + 1)*stride)]
                    test_X = self.full_X[i*stride: (i + 1)*stride, :]
                    test_Y = self.full_Y[i*stride: (i + 1)*stride, :]
                else:
                    train_X = self.full_X[:i*stride]
                    train_Y = self.full_Y[:i*stride]
                    test_X = self.full_X[i * stride:]
                    test_Y = self.full_Y[i * stride:]
                self.train_model.optimizer.zero_grad()
                predicted_y = self.train_model(train_X)
                temp_loss = self.train_model.compute_loss(predicted_y, train_Y)
                temp_loss.backward()
                self.train_model.optimizer.step()

                with torch.no_grad():
                    predicted_y_test = self.train_model(test_X)
                    test_loss += self.train_model.compute_loss(predicted_y_test, test_Y)
            if epoch % 200 == 0:
                self.log_factory.Slog(MessageAttribute.EInfo, sentences="Epoch={}, while loss={}".format(epoch, test_loss))
                computed_losses.append(test_loss.detach().clone())
        model_evaluation(computed_losses, epoch_step=200)

    def kill(self):
        self.log_factory.kill()