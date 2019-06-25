#%%
import netket as nk
import numpy as np
import pylab
import os
import json
import pandas as pd
import scipy.stats as sp

#%%
class NetKetOptimizer(object):
    """
    Neural Quantum States MaxCut problem.
    """

    def __init__(self, edgelist):
        """
        Initialize the graph.
        :param edgelist:
        """

        self.nk_graph = nk.graph.CustomGraph(edgelist)
        self.nk_hilbert = (nk
                           .hilbert
                           .CustomHilbert(graph=self.nk_graph, local_states=[-1, 1]))
        sz_sz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.nk_operator = nk.operator.GraphOperator(self.nk_hilbert, bondops=[sz_sz])
        self.nk_machine = nk.machine.RbmSpin(hilbert=self.nk_hilbert, alpha=2)
        self.nk_machine.init_random_parameters(sigma=0.1)
        self.nk_sampler = (nk
                           .sampler
                           .MetropolisExchange(machine=self.nk_machine,
                                               graph=self.nk_graph))
        self.nk_op = nk.optimizer.Momentum(0.001, 0.9)
        self.nk_fitter = nk.variational.Vmc(
            hamiltonian=self.nk_operator,
            sampler=self.nk_sampler,
            optimizer=self.nk_op,
            n_samples=1000)

    def run(self, n_iter=1500, prefix="learning_log"):
        """
        Run the process.
        :param n_iter: number of iterations.
        :param prefix: prefix for output files
        :return:
        """
        self.lr_prefix = prefix
        self.nk_fitter.run(prefix, n_iter)

    def get_result(self, exact, prefix="outdir"):
        """
        Get the result
        :param exact: exact solution
        :param prefix: output directory prefix
        :return:
        """

        iters = []
        energy_mean = []
        energy_sigma = []
        variance_mean = []
        variance_sigma = []

        input_file = os.path.join(self.lr_prefix, ".log")

        with open(input_file) as f:
            data = json.load(f)

            for iteration in data["Output"]:
                iters.append(iteration["Iteration"])
                energy_mean.append(iteration["Energy"]["Mean"])
                energy_sigma.append(iteration["Energy"]["Sigma"])
                variance_mean.append(iteration["EnergyVariance"]["Mean"])
                variance_sigma.append(iteration["EnergyVariance"]["Sigma"])

        p95 = sp.distributions.norm().ppf(0.975)

        results_df = pd.DataFrame(
            dict(
                iter=iters,
                e=energy_mean,
                e_std=energy_sigma,
                e_err95=[e * p95 for e in energy_sigma],
                e_var=variance_mean,
                e_var_std=variance_sigma,
                e_var_err95=[e * p95 for e in variance_sigma]
            )
        )

        out_path_csv = os.path.join(prefix, "history.csv")
        results_df.to_csv(out_path_csv, index=False)

        num_edges = len(self.nk_graph.edges)

        f, ax = pylab.subplots(nrows=1, ncols=2, figsize=(16, 5))
        ax[0].errorbar(results_df["iter"],
                       -(results_df["e"] - num_edges) / 2,
                       yerr=results_df["e_err95"],
                       label="Learning curve",
                       color="red")
        ax[0].plot(results_df["iter"],
                   np.ones(results_df.shape[0]) * exact,
                   "--", label="Exact solution")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("CutSize")
        ax[0].set_title("CutSize by iterations")
        ax[0].legend()

        ax[1].errorbar(
            results_df["iter"],
            results_df["e_var"],
            yerr=results_df["e_var_err95"],
            color="red"
        )
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Variance")
        ax[1].set_title("Variance of energy by iterations")

        f.tight_layout()

        out_path = os.path.join(prefix, "resultPlot.png")

        f.savefig(out_path, dpi=300)



