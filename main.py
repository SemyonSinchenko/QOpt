from src.NetketOptimizer import NetKetOptimizer
import numpy as np

if __name__ == "__main__":
    loaded_matrix = np.loadtxt("data/g05_60.0", skiprows=1, dtype=np.int32)
    edgelist = [[loaded_matrix[i, 0] - 1, loaded_matrix[i, 1] - 1]
                for i in range(loaded_matrix.shape[0])]
    opt = NetKetOptimizer(edgelist)

    print("Start optimizing process...")
    opt.run(2000, "/mountV/volume/learning_log")
    print("Done.")

    print("Save the results...")
    opt.get_result(536, "/mountV/volume/")
    print("Done. Finish. Success.")
