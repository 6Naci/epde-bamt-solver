import numpy as np
import pandas as pd
import torch
import scipy.io
import json
from TEDEouS import config


def wave_equation():
    """
        Load data
        Synthetic data from wolfram:

        WE = {D[u[x, t], {t, 2}] - 1/25 ( D[u[x, t], {x, 2}]) == 0}
        bc = {u[0, t] == 0, u[1, t] == 0};
        ic = {u[x, 0] == 10000 Sin[1/10 x (x - 1)]^2, Evaluate[D[u[x, t], t] /. t -> 0] == 1000 Sin[1/10  x (x - 1)]^2}
        NDSolve[Flatten[{WE, bc, ic}], u, {x, 0, 1}, {t, 0, 1}]
    """

    mesh = 60

    df = pd.read_csv(f'data/wave_equation/wolfram_sln/wave_sln_{mesh}.csv', header=None)
    data = df.values
    data = np.transpose(data)  # x1 - t (axis Y), x2 - x (axis X)

    t = np.linspace(0, 1, mesh + 1)
    x = np.linspace(0, 1, mesh + 1)
    grid = np.meshgrid(t, x, indexing='ij')

    param = [x, t]

    bconds = False  # if there are no boundary conditions
    """
    Preparing boundary conditions (BC)

    bnd=torch.Tensor of a boundary n-D points where n is the problem dimensionality
    
    bop=dictionary in the form of 
    'С * u{power: 1.0} * d^2u/dx1^2{power: 1.0}': 
            {
                'coeff': С ,
                'vars_set': [[None], [0, 0]],
                'power_set': [1.0, 1.0] 
            },

    bndval=torch.Tensor prescribed values at every point in the boundary
    """

    x_c = torch.from_numpy(x)
    t_c = torch.from_numpy(t)

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    bop1 = None

    # u(0,x)= 10000*sin[1/10 x*(x - 1)]^2
    bndval1 = 10000 * torch.sin((0.1 * bnd1[:, 0] * (bnd1[:, 0] - 1)) ** 2)

    # Initial conditions at t=0
    bnd2 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    # d/dt
    bop2 = {
        '1*du/dt**1':
            {
                'coefficient': 1,
                'du/dt': [1],
                'pow': 1
            }
    }

    # du/dt = 1000*sin[1/10 x*(x - 1)]^2
    bndval2 = 1000 * torch.sin((0.1 * bnd2[:, 0] * (bnd2[:, 0] - 1)) ** 2)

    # Boundary conditions at x=0
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t_c).float()

    bop3 = None

    # u(0,t)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t_c).float()

    bop4 = None
    # u(1,t)=0
    bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], [bnd3, bop3, bndval3], [bnd4, bop4, bndval4]]

    noise = False
    variance_arr = [0.001] if noise else [0]

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "eq_search_iter": 100
        },
        "set_memory_properties": {
            "mem_for_cache_frac": 10
        },
        "set_moeadd_params": {
            "population_size": 10,  #
            "training_epochs": 5
        },
        "Cache_stored_tokens": {
            "token_type": "grid",
            "token_labels": ["t", "x"],
            "params_ranges": {"power": (1, 1)},
            "params_equality_ranges": None
        },
        "fit": {
            "max_deriv_order": (2, 2),
            "boundary": 15,  #
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": 1,
            "eq_sparsity_interval": (1e-8, 5.0),  #
            "deriv_method": "poly",
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "glob_epde": {
            "test_iter_limit": 1,  # how many times to launch algorithm (one time - 2-3 equations)
            "variance_arr": variance_arr,
            "save_result": True,
            "load_result": False
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 3,
            "lambda": 0.01,
            "plot": False,
            "save_equations": True,
            "load_equations": False
        },
        "params": {
            "init_nodes": ['d^2u/dx2^2{power: 1.0}']
        }
    }

    solver_config = {
        "glob_solver": {
            "mode": "mat"
        },
        "Cache": {
            "use_cache": True,
            "save_always": True,
        }
    }

    ebs_config = {**epde_config, **bamt_config, **solver_config}

    path = 'data/wave_equation/'

    with open(f'{path}ebs_config.json', 'w') as fp:
        json.dump(ebs_config, fp)

    cfg_ebs = config.Config(f'{path}ebs_config.json')

    return data, grid, cfg_ebs, param, bconds


def burgers_equation():
    """
        Load data from github
        https://github.com/urban-fasel/EnsembleSINDy
        https://github.com/urban-fasel/EnsembleSINDy/blob/main/PDE-FIND/datasets/burgers.mat
    """
    mat = scipy.io.loadmat('burgers.mat')
    data = mat['u']
    t = mat['t']
    x = mat['x']
    # Create mesh
    grid = np.meshgrid(t, x, indexing='ij')

    param = [x, t]

    bconds = False  # if there are no boundary conditions

    """YOUR CODE HERE"""

    epde_config = """YOUR CODE HERE"""
    bamt_config = """YOUR CODE HERE"""
    solver_config = """YOUR CODE HERE"""

    ebs_config = {**epde_config, **bamt_config, **solver_config}

    path = """YOUR CODE HERE"""

    with open(f'{path}ebs_config.json', 'w') as fp:
        json.dump(ebs_config, fp)

    cfg_ebs = config.Config(f'{path}ebs_config.json')

    return data, grid, cfg_ebs, param, bconds
