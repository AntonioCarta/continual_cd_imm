import torch
import torch as th
import torch.nn as nn
from sklearn.cluster import kmeans_plusplus
from torch_dpmm.prob_utils.constraints import *
from torch_dpmm import GaussianDPMM


class CLGaussianDPMM(GaussianDPMM):

    def __init__(self, K, D, alphaDP, tau0, c0, n0, B0, is_diagonal):
        # we ensure that prior and posterior have the same shape, so we can use posterior as a prior in the next task
        super(CLGaussianDPMM, self).__init__(K, D, alphaDP, tau0, c0, n0, B0, is_diagonal, is_diagonal)

    @th.no_grad()
    def set_posterior_as_prior(self, component_mask):
        u, v, tau, c, n, B = self.get_var_params()

        self.u0 = self.u0.contiguous()
        self.v0 = self.v0.contiguous()
        self.tau0 = self.tau0.contiguous()
        self.c0 = self.c0.contiguous()
        self.n0 = self.n0.contiguous()
        self.B0 = self.B0.contiguous()

        self.u0[component_mask] = u[component_mask]
        self.v0[component_mask] = v[component_mask]
        self.tau0[component_mask] = tau[component_mask]
        self.c0[component_mask] = c[component_mask]
        self.n0[component_mask] = n[component_mask]
        self.B0[component_mask] = B[component_mask]

        self.__update_computation_fucntion__()

    def init_var_params(self, x=None, component_mask=None, variance_init_value=1):
        if component_mask is None:
            component_mask = th.ones(self.K, dtype=th.bool)

        K = th.sum(component_mask).item()

        u, v, tau, c, n, B = self.get_var_params()

        u[component_mask] = 1
        v[component_mask] = 1
        c[component_mask] = 1
        n[component_mask] = self.D + 2

        # var params of emission
        if x is not None:
            x_np = x.detach().numpy()
            # initialisation makes the difference: we should cover the input space
            mean_np, _ = kmeans_plusplus(x_np, K)
            tau[component_mask] = th.tensor(mean_np)
            B_eye_val = 2 * th.var(x) * th.ones(K, self.D)
        else:
            tau[component_mask] = 0
            B_eye_val = variance_init_value * th.ones(K, self.D)

        B[component_mask] = B_eye_val if self.is_diagonal else th.diag_embed(B_eye_val)

        self.__set_var_params__(u, v, tau, c, n, B)


class CD_IMM(nn.Module):

    def __init__(self, num_classes, K, D, use_smart_init=False, variance_init_value=None, **prior_params):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        self.D = D
        self.dpmm_list = nn.ModuleList([CLGaussianDPMM(K=K, D=D,  **prior_params) for _ in range(num_classes)])
        self.tot_counting = th.zeros(num_classes, K)
        self.current_task_counting = th.zeros(num_classes, K)
        self.use_smart_init = use_smart_init
        self.variance_init_value = variance_init_value  # the value to use to initialise variance var params

    def forward(self, x, y):
        elbo_tot = 0
        for c in range(self.num_classes):
            mask = y==c
            if th.any(mask):
                r, elbo = self.dpmm_list[c](x[mask])
                elbo_tot += elbo
                self.current_task_counting[c] += th.sum(r, 0)

        return elbo_tot

    @torch.no_grad()
    def predict(self, x):
        BS = x.shape[0]
        loglike = th.zeros(BS, self.num_classes)
        # assignment = th.zeros(BS, self.num_class, self.K)
        for c in range(self.num_classes):
            loglike[:, c] = self.dpmm_list[c].get_expected_log_likelihood(x)
            # assignment[mask, c, :] = pi

        return th.argmax(loglike, -1)

    def start_new_epoch(self):
        th.zero_(self.current_task_counting)

    def start_new_task(self, x, y):
        # the posterior becomes priors and we initialise not used components near data
        self.tot_counting += self.current_task_counting
        for c in range(self.num_classes):
            used_comp = (self.tot_counting[c] / th.sum(self.tot_counting[c])) > 0.01 # components with more thant 1% of the data
            self.dpmm_list[c].set_posterior_as_prior(used_comp)

            x_to_consider = None
            if self.use_smart_init:
                mask = y == c
                x_to_consider = x[mask]

            self.dpmm_list[c].init_var_params(x_to_consider, th.logical_not(used_comp),
                                              variance_init_value=self.variance_init_value)
