#%%
import pylab
import networkx as nx
import numpy as np

#%%
g05_60 = np.loadtxt("data/g05_60.0", skiprows=0, dtype=np.int32)
g05_100 = np.loadtxt("data/g05_100.0", skiprows=0, dtype=np.int32)

G60 = nx.convert.from_edgelist(
    [(g05_60[i, 0] - 1, g05_60[i, 1] - 1) for i in range(g05_60.shape[0])]
)

G100 = nx.convert.from_edgelist(
    [(g05_100[i, 0] - 1, g05_100[i, 1] - 1) for i in range(g05_100.shape[0])]
)

#%%
f60 = pylab.figure(figsize=(14, 14))
nx.drawing.draw(G60, nx.drawing.kamada_kawai_layout(G60))
f60.savefig("data/60vertexG.png", dpi=300)

#%%
f100 = pylab.figure(figsize=(14, 14))
nx.drawing.draw(G100, nx.drawing.kamada_kawai_layout(G100))
f100.savefig("data/100vertexG.png", dpi=300)
#%%