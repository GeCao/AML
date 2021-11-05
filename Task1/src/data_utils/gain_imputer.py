import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class GAINImputer:
    def __init__(self, core_management):
        self.core_management = core_management
        self.device = self.core_management.device

        self.data_set = None

        self.mini_batch_size = 128
        self.hint_rate = 0.9
        self.alpha = 10
        self.train_rate = 0.8
        self.Missing = None

        self.nrows = None
        self.input_dimension = None
        self.hidden_dimension_1 = None
        self.hidden_dimension_2 = None

        self.Train_No = None
        self.Test_No = None
        self.trainX = None
        self.testX = None
        self.trainM = None
        self.testM = None

        self.optimizer_D = None
        self.optimizer_G = None

        self.initialized = False

    def initialization(self, np_data):
        self.data_set = np_data

        self.nrows = self.data_set.shape[0]
        self.input_dimension = self.data_set.shape[1]
        self.hidden_dimension_1 = self.input_dimension
        self.hidden_dimension_2 = self.input_dimension

        # Normalization (0 to 1)
        Min_Val = np.zeros(self.input_dimension)
        Max_Val = np.zeros(self.input_dimension)
        for i in range(self.input_dimension):
            Min_Val[i] = np.min(self.data_set[:, i])
            self.data_set[:, i] = self.data_set[:, i] - np.nanmin(self.data_set[:, i])
            Max_Val[i] = np.max(self.data_set[:, i])
            self.data_set[:, i] = self.data_set[:, i] / (np.nanmax(self.data_set[:, i]) + 1e-6)
            if np.isnan(np.nanmax(self.data_set[:, i])) or np.isnan(np.nanmax(self.data_set[:, i])):
                print("all nan in this column")

        self.Missing = np.where(np.isnan(self.data_set), 0.0, 1.0)
        self.data_set = np.where(np.isnan(self.data_set), 0.0, self.data_set)

        idx = np.random.permutation(self.nrows)
        self.Train_No = int(self.nrows * self.train_rate)
        self.Test_No = self.nrows - self.Train_No

        # Train / Test Features
        self.trainX = self.data_set[idx[:self.Train_No], :]
        self.testX = self.data_set[idx[self.Train_No:], :]

        # Train / Test Missing Indicators
        self.trainM = self.Missing[idx[:self.Train_No], :]
        self.testM = self.Missing[idx[self.Train_No:], :]

        self.initialized = True

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(self.input_dimension / 2.)
        return np.random.normal(size=size, scale=xavier_stddev)

    def sample_M(self, m, n, p):
        # Hint Vector Generation
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    def train(self):
        # 1. Discriminator
        self.D_W1 = torch.tensor(self.xavier_init([self.input_dimension * 2, self.hidden_dimension_1]),
                                 requires_grad=True, device=self.device)  # Data + Hint as inputs
        self.D_b1 = torch.tensor(np.zeros(shape=[self.hidden_dimension_1]),
                                 requires_grad=True, device=self.device)

        self.D_W2 = torch.tensor(self.xavier_init([self.hidden_dimension_1, self.hidden_dimension_2]),
                                 requires_grad=True, device=self.device)
        self.D_b2 = torch.tensor(np.zeros(shape=[self.hidden_dimension_2]),
                                 requires_grad=True, device=self.device)

        self.D_W3 = torch.tensor(self.xavier_init([self.hidden_dimension_2, self.input_dimension]),
                                 requires_grad=True, device=self.device)
        self.D_b3 = torch.tensor(np.zeros(shape=[self.input_dimension]),
                                 requires_grad=True, device=self.device)
        theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # 2. Generator
        self.G_W1 = torch.tensor(self.xavier_init([self.input_dimension * 2, self.hidden_dimension_1]),
                                 requires_grad=True,
                                 device=self.device)  # Data + Mask as inputs (Random Noises are in Missing Components)
        self.G_b1 = torch.tensor(np.zeros(shape=[self.hidden_dimension_1]),
                                 requires_grad=True, device="cuda")

        self.G_W2 = torch.tensor(self.xavier_init([self.hidden_dimension_1, self.hidden_dimension_2]),
                                 requires_grad=True, device="cuda")
        self.G_b2 = torch.tensor(np.zeros(shape=[self.hidden_dimension_2]),
                                 requires_grad=True, device="cuda")

        self.G_W3 = torch.tensor(self.xavier_init([self.hidden_dimension_2, self.input_dimension]),
                                 requires_grad=True, device="cuda")
        self.G_b3 = torch.tensor(np.zeros(shape=[self.input_dimension]),
                                 requires_grad=True, device="cuda")
        theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

        self.optimizer_D = torch.optim.Adam(params=theta_D)
        self.optimizer_G = torch.optim.Adam(params=theta_G)

        # %% Start Iterations
        for it in tqdm(range(5000)):

            # %% Inputs
            mb_idx = self.sample_idx(self.Train_No, self.mini_batch_size)
            X_mb = self.trainX[mb_idx, :]

            Z_mb = self.sample_Z(self.mini_batch_size, self.input_dimension)
            M_mb = self.trainM[mb_idx, :]
            H_mb1 = self.sample_M(self.mini_batch_size, self.input_dimension, 1 - self.hint_rate)
            H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            X_mb = torch.tensor(X_mb, device=self.device)
            M_mb = torch.tensor(M_mb, device=self.device)
            H_mb = torch.tensor(H_mb, device=self.device)
            New_X_mb = torch.tensor(New_X_mb, device=self.device)

            self.optimizer_D.zero_grad()
            D_loss_curr = self.discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb,
                                                                                       H=H_mb)
            G_loss_curr.backward()
            self.optimizer_G.step()

            # %% Intermediate Losses
            if it % 100 == 0:
                self.core_management.log_factory.InfoLog('Iter: {}'.format(it))
                self.core_management.log_factory.InfoLog(
                    'Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
                self.core_management.log_factory.InfoLog('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
        New_X = self.Missing * self.data_set + (1 - self.Missing) * self.sample_Z(self.nrows, self.input_dimension)
        with torch.no_grad():
            New_X = torch.tensor(New_X, device=self.device)
            M = torch.tensor(self.Missing, device=self.device)
            G_sample = self.generator(New_X, M)
            Hat_New_X = New_X * M + G_sample * (1 - M)
            return Hat_New_X.cpu().numpy()

    def generator(self, new_x, m):
        inputs = torch.cat(dim=1, tensors=[new_x, m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2) + self.G_b2)
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3) + self.G_b3)  # [0,1] normalized Output
        return G_prob

    def discriminator(self, new_x, h):
        inputs = torch.cat(dim=1, tensors=[new_x, h])  # Hint + Data Concatenate
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = torch.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
        return D_prob

    def sample_Z(self, m, n):
        return np.random.uniform(0., 0.01, size=[m, n])

    def sample_idx(self, m, n):
        # Mini-batch generation
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    def discriminator_loss(self, M, New_X, H):
        # Generator
        G_sample = self.generator(New_X, M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = self.discriminator(Hat_New_X, H)

        # %% Loss
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def generator_loss(self, X, M, New_X, H):
        # %% Structure
        # Generator
        G_sample = self.generator(New_X, M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = self.discriminator(Hat_New_X, H)

        # %% Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

        G_loss = G_loss1 + self.alpha * MSE_train_loss

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def test_loss(self, X, M, New_X):
        # %% Structure
        # Generator
        G_sample = self.generator(New_X, M)

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return MSE_test_loss, G_sample
