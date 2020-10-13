import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import scipy.sparse as sp
import numpy as np
from utils import accuracy, sparse_matrix_to_sparse_tensor
from copy import deepcopy


def sparse_dense_mul(a, b, c):
    i = a._indices()[1]
    j = a._indices()[0]
    v = a._values()
    newv = (b[i] + c[j]).squeeze()
    newv = torch.exp(F.leaky_relu(newv))

    new = torch.sparse.FloatTensor(a._indices(), newv, a.size())
    return new


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, activation=None, dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll = nn.Linear(in_features, out_features).cuda()
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, adj, dropout=0.5):
        x = self.ll(x)
        x = torch.spmm(adj, x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)
        return x


class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features, activation=None, dropout=False):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll = nn.Linear(in_features, out_features).cuda()

        self.activation = activation
        self.dropout = dropout

    def forward(self, x, adj, dropout=0):
        x = self.ll(x)

        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)
        return x


class MLP(nn.Module):
    def __init__(self, num_layers, num_features):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    MLPLayer(num_features[i], num_features[i + 1], activation=F.elu, dropout=True).cuda())
            else:
                self.layers.append(MLPLayer(num_features[i], num_features[i + 1]).cuda())

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)
        return x


class GCN(nn.Module):
    def __init__(self, num_layers, num_features):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    # GraphConvolution(num_features[i], num_features[i + 1], activation=F.elu, dropout=True).cuda())
                    GraphConvolution(num_features[i], num_features[i + 1], activation=F.relu, dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i], num_features[i + 1]).cuda())

    def forward(self, x, adj, dropout=0.5):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)

        return x


class MLP_norm(nn.Module):
    def __init__(self, num_layers, num_features):
        super(MLP_norm, self).__init__()
        self.MLP = MLP(num_layers, num_features)

    def forward(self, x, adj, dropout=0):
        x = F.relu(x + 1.8) - 1.8
        x = 1.8 - F.relu(1.8 - x)
        x = self.MLP(x, adj, dropout)
        return x


class GCN_norm(nn.Module):
    def __init__(self, num_layers, num_features):
        super(GCN_norm, self).__init__()
        self.GCN = GCN(num_layers, num_features)
        self.lr = 0.005
        self.weight_decay = 5e-5  # 0.009 #5e-4
        self.device = 'cuda'
        self.loss = torch.nn.CrossEntropyLoss()
        self.ln = nn.LayerNorm(100).cuda()
        # self.loss = F.nll_loss

    def forward(self, x, adj, dropout=0):
        x = F.relu(x + 1.8) - 1.8
        x = 1.8 - F.relu(1.8 - x)
        # added layer norm
        x = self.ln(x)
        x = self.GCN(x, adj, dropout)
        return x

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=100, verbose=False, patience=50):
        self.LOSSES = []
        self.LOSSES_VAL = []
        self.ACCURACIES = []

        self.adj_norm = sparse_matrix_to_sparse_tensor(GCNadj(adj)).to(self.device)
        self.adj_t = sparse_matrix_to_sparse_tensor(adj.tocoo()).to(self.device)
        self.features = torch.FloatTensor(features).to(self.device)
        self.labels = torch.LongTensor(labels).to(self.device)

        assert patience < train_iters, "Patience should be smaller than train iterations"
        self._train_with_early_stopping(self.labels, idx_train, idx_val, train_iters, patience, verbose)

        return self.LOSSES, self.LOSSES_VAL, self.ACCURACIES

    def adj_loss(self, adj_t, output):
        '''
        adj_t is a tensor (sparse) and output is also a tensor (dense of size NxD but N is large, D is small enough)
        torch.norm(self.adj_t - torch.mm(output, torch.transpose(output, 0, 1)))
        '''
        pass

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 10000

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output[idx_train], labels[idx_train])  # + 0.1 *

            self.LOSSES.append(loss_train.cpu().item())
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            loss_val = self.loss(output[idx_val], labels[idx_val])
            self.LOSSES_VAL.append(loss_val.item())
            acc_val = accuracy(output[idx_val], labels[idx_val])
            self.ACCURACIES.append(acc_val)

            print('Epoch {}, training loss: {:.4f}, validation accuracy: {:.4f}'.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)
        torch.save(weights, "weights_relu")


class MyModel(nn.Module):
    def __init__(self, model_name):
        super(MyModel, self).__init__()
        self.model_name = model_name
        gcn_dims = [100, 256, 128, 64]
        self.gcn_norm = GCN_norm(len(gcn_dims) - 1, gcn_dims)
        mlp_dims = [100, 128, 64]
        self.mlp_norm = MLP_norm(len(mlp_dims) - 1, mlp_dims)
        self.ln1 = nn.LayerNorm(64).cuda()
        self.ln2 = nn.LayerNorm(64).cuda()
        self.ll1 = nn.Linear(128, 64).cuda()
        self.ll2 = nn.Linear(64, 18).cuda()

        self.lr = 0.005
        self.weight_decay = 5e-5  # 0.009 #5e-4
        self.device = 'cuda'
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, adj, dropout=0.1):
        x1 = self.gcn_norm(x, adj, dropout)
        x1 = self.ln1(x1)
        x2 = self.mlp_norm(x, adj, dropout)
        x2 = self.ln2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = F.elu(self.ll1(x))
        x = self.ll2(x)
        return x

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output[idx_train], labels[idx_train])

            self.LOSSES.append(loss_train.cpu().item())
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            loss_val = self.loss(output[idx_val], labels[idx_val])
            self.LOSSES_VAL.append(loss_val.item())
            acc_val = accuracy(output[idx_val], labels[idx_val])
            self.ACCURACIES.append(acc_val)

            # if verbose and i % 10 == 0:
            print('Epoch {}, training loss: {:.4f}, validation accuracy: {:.4f}'.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        print(f'Storing back the weights of best loss val {best_loss_val} and acc val {best_acc_val}')
        self.load_state_dict(weights)

        torch.save(weights, self.model_name)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=100, verbose=False, patience=50):
        self.LOSSES = []
        self.LOSSES_VAL = []
        self.ACCURACIES = []

        self.adj_norm = sparse_matrix_to_sparse_tensor(GCNadj(adj)).to(self.device)
        self.features = torch.FloatTensor(features).to(self.device)
        self.labels = torch.LongTensor(labels).to(self.device)

        assert patience < train_iters, "Patience should be smaller than train iterations"
        self._train_with_early_stopping(self.labels, idx_train, idx_val, train_iters, patience, verbose)

        return self.LOSSES, self.LOSSES_VAL, self.ACCURACIES


def GCNadj(adj):
    adj = sp.eye(adj.shape[0]) + adj
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj
