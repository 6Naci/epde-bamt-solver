import numpy as np
import pandas as pd
import torch
import scipy.io


def  wave_equation(grid_res, title='wave_equation'):
    """
        Load data
        Synthetic data from wolfram:

        WE = {D[u[x, t], {t, 2}] - 1/25 ( D[u[x, t], {x, 2}]) == 0}
        bc = {u[0, t] == 0, u[1, t] == 0};
        ic = {u[x, 0] == 10000 Sin[1/10 x (x - 1)]^2, Evaluate[D[u[x, t], t] /. t -> 0] == 1000 Sin[1/10  x (x - 1)]^2}
        NDSolve[Flatten[{WE, bc, ic}], u, {x, 0, 1}, {t, 0, 1}]
    """

    df = pd.read_csv(f'data/{title}/wolfram_sln/wave_sln_{grid_res}.csv', header=None)
    u = df.values
    u = np.transpose(u)  # x1 - t (axis Y), x2 - x (axis X)

    t = np.linspace(0, 1, grid_res + 1)
    x = np.linspace(0, 1, grid_res + 1)
    params = [t, x]

    grid = torch.cartesian_prod(torch.from_numpy(t), torch.from_numpy(x)).float()

    return u, grid


def burgers_equation():
    """
        Load data from github
        https://github.com/urban-fasel/EnsembleSINDy
        https://github.com/urban-fasel/EnsembleSINDy/blob/main/PDE-FIND/datasets/burgers.mat
    """
    mat = scipy.io.loadmat('burgers.mat')
    u = mat['u']
    t = mat['t']
    x = mat['x']
    # Create mesh
    grid = np.meshgrid(t, x, indexing='ij')

    return u, grid


