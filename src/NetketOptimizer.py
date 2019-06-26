#%%
import netket as nk
import numpy as np
import pylab
import os
import json
import pandas as pd
import scipy.stats as sp
import sys

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
        sys.stdout.write("Created graph with {} vertices and {} edges".format(
            self.nk_graph.n_sites, len(self.nk_graph.edges)))
        self.nk_hilbert = (nk
                           .hilbert
                           .CustomHilbert(graph=self.nk_graph, local_states=[-1, 1]))
        sz_sz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.nk_operator = nk.operator.GraphOperator(self.nk_hilbert, bondops=[sz_sz])

        # self.nk_machine = nk.machine.RbmSpin(hilbert=self.nk_hilbert, alpha=1)
        self.nk_machine = nk.machine.FFNN(
            hilbert=self.nk_hilbert,
            layers=(
                nk.layer.FullyConnected(input_size=self.nk_graph.n_sites, output_size=30, use_bias=True),
                nk.layer.FullyConnected(input_size=30, output_size=20, use_bias=True),
                nk.layer.FullyConnected(input_size=20, output_size=10, use_bias=True),
                nk.layer.FullyConnected(input_size=10, output_size=10, use_bias=True),
                nk.layer.Lncosh(input_size=10),
                nk.layer.SumOutput(input_size=10)
            )
        )
        self.nk_machine.init_random_parameters(sigma=0.1)
        self.nk_sampler = (nk
                           .sampler
                           .MetropolisExchange(machine=self.nk_machine,
                                               graph=self.nk_graph))

        sys.stdout.write("Cretaed optimizer.")
        sys.stdout.write("Current state:")
        for i, v in enumerate(self.nk_sampler.visible):
            sys.stdout.write("{}th spin orientation is {}".format(i, v))

        self.nk_op = nk.optimizer.Sgd(0.01, 0.02, 0.995)
        self.nk_fitter = nk.variational.Vmc(
            hamiltonian=self.nk_operator,
            sampler=self.nk_sampler,
            optimizer=self.nk_op,
            n_samples=5000,
            discarded_samples=3500,
            diag_shift=0.03,
            use_iterative=True
        )

    def run(self, n_iter=600, prefix="learning_log"):
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
        acceptance = []

        input_file = self.lr_prefix + ".log"

        with open(input_file) as f:
            data = json.load(f)

            for iteration in data["Output"]:
                iters.append(iteration["Iteration"])
                acceptance.append(iteration["Acceptance"])
                energy_mean.append(iteration["Energy"]["Mean"])
                energy_sigma.append(iteration["Energy"]["Sigma"])
                variance_mean.append(iteration["EnergyVariance"]["Mean"])
                variance_sigma.append(iteration["EnergyVariance"]["Sigma"])

        p95 = sp.distributions.norm().ppf(0.975)

        results_df = pd.DataFrame(
            dict(
                iter=iters,
                accept=acceptance,
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

        f, ax = pylab.subplots(nrows=1, ncols=3, figsize=(20, 5))
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
        ax[0].grid()
        ax[0].legend()
        ax[0].text(
            results_df["iter"].iloc[-1000],
            -(results_df["e"].iloc[-1] - num_edges) / 2 + 5,
            "Last value is {.2f}".format(-(results_df["e"].iloc[-1] - num_edges) / 2))

        ax[1].errorbar(
            results_df["iter"],
            results_df["e_var"],
            yerr=results_df["e_var_err95"],
            color="red"
        )
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Variance")
        ax[1].set_title("Variance of energy by iterations")
        ax[1].grid()

        ax[2].plot(results_df["iter"], results_df["accept"], ".-")
        ax[2].set_xlabel("Iteration")
        ax[2].set_ylabel("Acceptance ratio")
        ax[2].set_title("Acceptance ration by iterations")
        ax[2].grid()

        f.tight_layout()

        out_path = os.path.join(prefix, "resultPlot.png")

        f.savefig(out_path, dpi=300)

        state_file_path = os.path.join(prefix, "lastState.txt")
        np.savetxt(state_file_path, self.nk_sampler.visible, ".1f")

        sys.stdout.write("Generate some samples...")
        samples = []
        for i in range(1000):
            self.nk_sampler.sweep()
            samples.append(self.nk_sampler.visible)

        last_state_file_path = os.path.join(prefix, "stateAdvanced_1000steps.txt")
        np.savetxt(last_state_file_path, np.array(samples), fmt="%.1f")



