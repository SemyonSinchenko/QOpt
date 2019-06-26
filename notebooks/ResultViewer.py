#%%
import os
import scipy.stats as sp
import pylab
import numpy as np
import json
import pandas as pd

#%%
iters = []
energy_mean = []
energy_sigma = []
variance_mean = []
variance_sigma = []
exact = 536

input_file = os.path.join("results", "learning_log.log")

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

num_edges = 884

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
ax[0].grid()

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
f.savefig(os.path.join("results", "learningCurve.png"))
f.show()

#%%