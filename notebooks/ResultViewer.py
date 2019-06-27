#%%
import os
import pylab
import numpy as np

#%%

NUM_EDGES = 2474
EXACT_SOLUTION = 1430
samples = np.loadtxt(os.path.join("results", "FFNN", "100vertexGraph","stateAdvanced_1000steps.txt"))

loaded_matrix = np.loadtxt("data/g05_100.0", skiprows=0, dtype=np.int32)
edgelist = [[loaded_matrix[i, 0] - 1, loaded_matrix[i, 1] - 1]
            for i in range(loaded_matrix.shape[0])]

#%%

def score(state, edges):
    r = -NUM_EDGES
    for e in edges:
        r += state[e[0]] * state[e[1]]

    return -r / 2

results = []
for i in range(samples.shape[0]):
    results.append(score(samples[i, :], edgelist))

results = np.array(results)

#%%

pylab.figure(figsize=(8, 4))
pylab.plot(np.arange(results.shape[0]), results, ".-", label="Results")
pylab.plot(np.arange(results.shape[0]),
           np.ones(results.shape[0]) * EXACT_SOLUTION, "--", label="Exact")
pylab.xlabel("Sample number")
pylab.ylabel("CutSize")
pylab.legend()
pylab.grid()
pylab.savefig(os.path.join("results", "FFNN", "100vertexGraph", "SamplesResults.png"))

#%%