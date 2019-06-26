from src.NetketOptimizer import NetKetOptimizer
import numpy as np
import sys

if __name__ == "__main__":
    loaded_matrix = np.loadtxt("/mountV/volume/QOpt/data/g05_60.0", skiprows=1, dtype=np.int32)
    edgelist = [[loaded_matrix[i, 0] - 1, loaded_matrix[i, 1] - 1]
                for i in range(loaded_matrix.shape[0])]
    opt = NetKetOptimizer(edgelist)

    sys.stdout.write("Start optimizing process...")
    opt.run(400, "/mountV/volume/learning_log")
    sys.stdout.write("Done.")

    sys.stdout.write("Save the results...")
    opt.get_result(536, "/mountV/volume/")
    sys.stdout.write("Done. Finish. Success.")
