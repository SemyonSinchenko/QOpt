from src.NetketOptimizer import NetKetOptimizer
import numpy as np
import sys
import pathlib
import os

if __name__ == "__main__":
    loaded_matrix = np.loadtxt(os.path.join(os.sep, "mountV", "volume", "QOpt", "data", "g05_60.0"), dtype=np.int32)
    edgelist = [[loaded_matrix[i, 0] - 1, loaded_matrix[i, 1] - 1]
                for i in range(loaded_matrix.shape[0])]
    opt = NetKetOptimizer(edgelist)

    sys.stdout.write("Start optimizing process...")
    pathlib.Path(os.path.join(os.sep, "mountV", "volume", "60vertexGraph")).mkdir(parents=True, exist_ok=True)
    opt.run(400, os.path.join(os.sep, "mountV", "volume", "60vertexGraph", "learning_log"))
    sys.stdout.write("Done.")

    sys.stdout.write("Save the results...")
    opt.get_result(536, os.path.join(os.sep, "mountV", "volume", "60vertexGraph"))
    sys.stdout.write("Done. Finish. Success.")


    loaded_matrix = np.loadtxt(os.path.join(os.sep, "mountV", "volume", "QOpt", "data", "g05_100.0"), dtype=np.int32)
    edgelist = [[loaded_matrix[i, 0] - 1, loaded_matrix[i, 1] - 1]
                for i in range(loaded_matrix.shape[0])]
    opt = NetKetOptimizer(edgelist, layers=[40, 40, 30, 10])

    sys.stdout.write("Start optimizing process...")
    pathlib.Path(os.path.join(os.sep, "mountV", "volume", "100vertexGraph")).mkdir(parents=True, exist_ok=True)
    opt.run(800, os.path.join(os.sep, "mountV", "volume", "100vertexGraph", "learning_log"))
    sys.stdout.write("Done.")

    sys.stdout.write("Save the results...")
    opt.get_result(1430, os.path.join(os.sep, "mountV", "volume", "100vertexGraph"))
    sys.stdout.write("Done. Finish. Success.")
