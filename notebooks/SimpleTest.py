#%%
import netket as nk
import networkx as nx
import numpy as np
import pylab

#%%
g05_60_1 = np.loadtxt("data/g05_60.0", skiprows=1, dtype=np.int32)

#%%
G = nx.convert.from_edgelist(
    [(g05_60_1[i, 0] - 1, g05_60_1[i, 1] - 1) for i in range(g05_60_1.shape[0])]
)
#%%
print(len(G.edges))
print(len(G.nodes))

#%%
nx.drawing.draw(G, nx.drawing.kamada_kawai_layout(G))
pylab.show()

#%%
nk_graph = nk.graph.CustomGraph(
    [[g05_60_1[i, 0], g05_60_1[i, 1]] for i in range(g05_60_1.shape[0])]
)


#%%
nk_hilbert = nk.hilbert.CustomHilbert(graph=nk_graph, local_states=[-1, 1])

#%%
sz_sz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
nk_operator = nk.operator.GraphOperator(nk_hilbert, bondops=[sz_sz])

#%%
# nk_machine = nk.machine.FFNN(
#     hilbert=nk_hilbert,
#     layers=(
#         nk.layer.FullyConnected(input_size=61, output_size=20, use_bias=True),
#         nk.layer.Lncosh(input_size=20),
#         nk.layer.SumOutput(input_size=20)
#     )
# )
nk_machine = nk.machine.RbmSpin(hilbert=nk_hilbert, alpha=2)
nk_machine.init_random_parameters(sigma=0.1)

#%%
nk_sampler = nk.sampler.MetropolisExchange(machine=nk_machine, graph=nk_graph)

#%%
nk_op = nk.optimizer.Momentum(0.001, 0.9)

#%%
nk_fitter = nk.variational.Vmc(
    hamiltonian=nk_operator,
    sampler=nk_sampler,
    optimizer=nk_op,
    n_samples=500
)

#%%
nk_fitter.run("g05_60_results", 10)

#%%