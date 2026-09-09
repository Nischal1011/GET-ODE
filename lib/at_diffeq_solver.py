'''
ODE solver for AT-LG-ODE.

Mirrors lib/diffeq_solver.py's DiffeqSolver/GraphODEFunc exactly (including using the memory-
efficient odeint_adjoint), with one addition: GraphODEFunc.forward passes t_local down into the
ODE net so it can query the AttentionTransport module for time-dependent edge weights w_ij(t).

The cached attention that w_ij(t) is built from is detached (see attention_transport.py), so
the encoder keeps training only through its original z0 (attention-pooling) path -- exactly
like plain LG-ODE -- and adjoint_params never needs to include anything beyond the ODE net's
own registered parameters (which already includes AttentionTransport's learnable lambda, since
it's a submodule of ode_func_net).
'''
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np


class ATDiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, args,
                 odeint_rtol=1e-3, odeint_atol=1e-4, device=torch.device("cpu")):
        super(ATDiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.n_balls

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

        self.rel_rec, self.rel_send = self.compute_rec_send()

    def compute_rec_send(self):
        off_diag = np.ones([self.num_atoms, self.num_atoms]) - np.eye(self.num_atoms)
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        rel_send = torch.FloatTensor(rel_send).to(self.device)
        return rel_rec, rel_send

    def forward(self, first_point, time_steps_to_predict, graph, backwards=False):
        ispadding = False
        if time_steps_to_predict[0] != 0:
            ispadding = True
            time_steps_to_predict = torch.cat((torch.zeros(1, device=time_steps_to_predict.device), time_steps_to_predict))

        n_traj_samples, n_traj, feature = first_point.size()[0], first_point.size()[1], first_point.size()[2]
        first_point_augumented = first_point.view(-1, self.num_atoms, feature)
        if self.args.augment_dim > 0:
            aug = torch.zeros(first_point_augumented.shape[0], first_point_augumented.shape[1], self.args.augment_dim).to(self.device)
            first_point_augumented = torch.cat([first_point_augumented, aug], 2)
            feature += self.args.augment_dim

        graph_augmented = torch.cat([graph for _ in range(n_traj_samples)], dim=0)

        rel_type_onehot = torch.FloatTensor(first_point_augumented.size(0), self.rel_rec.size(0),
                                             self.args.edge_types).to(self.device)
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, graph_augmented.view(first_point_augumented.size(0), -1, 1), 1)

        self.ode_func.set_graph(rel_type_onehot, self.rel_rec, self.rel_send, self.args.edge_types)
        self.ode_func.transport.set_adjacency(graph, n_traj_samples=n_traj_samples)

        pred_y = odeint(self.ode_func, first_point_augumented, time_steps_to_predict,
                         rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)

        if ispadding:
            pred_y = pred_y[1:, :, :, :]
            time_steps_to_predict = time_steps_to_predict[1:]

        pred_y = pred_y.view(time_steps_to_predict.size(0), -1, pred_y.size(3))
        pred_y = pred_y.permute(1, 0, 2)
        pred_y = pred_y.view(n_traj_samples, n_traj, -1, feature)

        assert (pred_y.size()[0] == n_traj_samples)
        assert (pred_y.size()[1] == n_traj)

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :, :-self.args.augment_dim]

        return pred_y

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot


class ATGraphODEFunc(nn.Module):
    def __init__(self, ode_func_net, transport, device=torch.device("cpu")):
        super(ATGraphODEFunc, self).__init__()
        self.device = device
        self.ode_func_net = ode_func_net
        self.transport = transport
        self.nfe = 0

    def forward(self, t_local, z, backwards=False):
        self.nfe += 1
        grad = self.ode_func_net(z, t_local=t_local)
        if backwards:
            grad = -grad
        return grad

    def set_graph(self, rec_type, rel_rec, rel_send, edge_types):
        for layer in self.ode_func_net.gcs:
            layer.base_conv.rel_type = rec_type
            layer.base_conv.rel_rec = rel_rec
            layer.base_conv.rel_send = rel_send
            layer.base_conv.edge_types = edge_types
        self.nfe = 0
