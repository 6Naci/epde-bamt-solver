import math
import os
import numpy as np
import pandas as pd
import torch
import scipy.io
import json
from TEDEouS import config
from default_configs import DEFAULT_CONFIG_EBS

config.default_config = json.loads(DEFAULT_CONFIG_EBS)


def example_equation():
    """
        path -> data -> parameters -> derivatives (optional) -> grid -> boundary conditions (optional) -> modules config (optional)
    """
    path = """YOUR CODE HERE"""
    data = """YOUR CODE HERE"""

    derives = None  # if there are no derivatives

    grid = """YOUR CODE HERE"""
    param = """YOUR CODE HERE"""

    bconds = False  # if there are no boundary conditions

    """
    Preparing boundary conditions (BC)

    For every boundary we define three items

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)

    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

    Meaning c1*u*d2u/dx2 has the form

    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}

    None is for function without derivatives

    bval=torch.Tensor prescribed values at every point in the boundary

    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], ...]
    """

    noise = False
    variance_arr = ["""YOUR CODE HERE"""] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": data.ndim - 1 # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
        }
    }

    epde_config = {"""YOUR CODE HERE"""}

    bamt_config = {"""YOUR CODE HERE"""}

    solver_config = {"""YOUR CODE HERE"""}

    config_modules = {**global_modules,
                      **epde_config,
                      **bamt_config,
                      **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, param, bconds


def wave_equation():
    """
        Load data
        Synthetic data from wolfram:

        WE = {D[u[x, t], {t, 2}] - 1/25 ( D[u[x, t], {x, 2}]) == 0}
        bc = {u[0, t] == 0, u[1, t] == 0};
        ic = {u[x, 0] == 10000 Sin[1/10 x (x - 1)]^2, Evaluate[D[u[x, t], t] /. t -> 0] == 1000 Sin[1/10  x (x - 1)]^2}
        NDSolve[Flatten[{WE, bc, ic}], u, {x, 0, 1}, {t, 0, 1}]
    """

    mesh = 70

    path = 'data/wave_equation/'
    df = pd.read_csv(f'{path}wolfram_sln/wave_sln_{mesh}.csv', header=None)
    data = df.values
    data = np.transpose(data)  # x1 - t (axis Y), x2 - x (axis X)

    derives = None

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
    bconds = [[bnd1, bndval1], [bnd2, bop2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]
    noise = False
    variance_arr = [0.001] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": 1,
            "variance_arr": variance_arr
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 15,  #
            "verbose_params": {"show_moeadd_epochs": True}
        },
        # "set_memory_properties": {
        #     "mem_for_cache_frac": 10
        # },
        "set_moeadd_params": {
            "population_size": 3,  #
            "training_epochs": 5
        },
        # "Cache_stored_tokens": {
        #     "token_type": "grid",
        #     "token_labels": ["t", "x"],
        #     "params_ranges": {"power": (1, 1)},
        #     "params_equality_ranges": None
        # },
        "fit": {
            "variable_names": ['u', ],
            "max_deriv_order": (2, 2),
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": 1,
            "eq_sparsity_interval": (1e-8, 5.0),  #
            "deriv_method": "poly",
            "deriv_method_kwargs": {"smooth": True},
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "glob_epde": {
            "test_iter_limit": 3,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False,
            "load_result": True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 35,
            "lambda": 0.0001,
            "plot": False,
            "save_result": False,
            "load_result": True
        },
        "params": {
            "init_nodes": ['d^2u/dx2^2{power: 1.0}']
        },
        "correct_structures": {
            "list_unique": ['d^2u/dx2^2{power: 1.0}', 'd^2u/dx1^2{power: 1.0}_r']
        }
    }

    img_dir = f'{path}wave_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": "NN",
            "reverse": True
        },
        "Cache": {
            "use_cache": True,
            "save_always": False,
            "cache_dir": f"{path}cache/"
        },
        "Optimizer": {
            "learning_rate": 1e-3,
            "lambda_bound": 100,
            "optimizer": "Adam"
        },
        "Plot": {
            "step_plot_print": False,
            "step_plot_save": False,
            "image_save_dir": img_dir,
        }
    }

    ebs_config = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}ebs_config.json', 'w') as fp:
        json.dump(ebs_config, fp)

    cfg_ebs = config.Config(f'{path}ebs_config.json')

    return data, grid, derives, cfg_ebs, param, bconds

# no changes
def burgers_equation():
    """
        Load data from github
        https://github.com/urban-fasel/EnsembleSINDy
        https://github.com/urban-fasel/EnsembleSINDy/blob/main/PDE-FIND/datasets/burgers.mat
    """
    path = "data/burgers_equation/"
    mat = scipy.io.loadmat(f'{path}burgers.mat')

    data = mat['u']
    data = np.transpose(data)
    t = np.ravel(mat['t'])
    x = np.ravel(mat['x'])

    derives = None
    dx = pd.read_csv(f'{path}wolfram_sln_derv/burgers_sln_dx_256.csv', header=None)
    d_x = dx.values
    d_x = np.transpose(d_x)

    dt = pd.read_csv(f'{path}wolfram_sln_derv/burgers_sln_dt_256.csv', header=None)
    d_t = dt.values
    d_t = np.transpose(d_t)

    dtt = pd.read_csv(f'{path}wolfram_sln_derv/burgers_sln_dtt_256.csv', header=None)
    d_tt = dtt.values
    d_tt = np.transpose(d_tt)

    # derives = np.zeros(shape=(data.shape[0], data.shape[1], 3))
    # derives[:, :, 0] = d_t
    # derives[:, :, 1] = d_tt
    # derives[:, :, 2] = d_x

    derives = np.zeros(shape=(data.shape[0], data.shape[1], 2))
    derives[:, :, 0] = d_t
    derives[:, :, 1] = d_x

    # Create mesh
    grid = np.meshgrid(t, x, indexing='ij')

    param = [x, t]

    bconds = False  # if there are no boundary conditions
    x_c = torch.from_numpy(x)
    t_c = torch.from_numpy(t)

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    bop1 = None
    # u(x, 0) = Piecewise
    bndval1 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval1.csv', header=None).values).reshape(-1)

    # Initial conditions at t=4
    bnd2 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([4], dtype=np.float64))).float()
    bop2 = None
    # u(x, 4) = Piecewise
    bndval2 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval1_2.csv', header=None).values).reshape(-1)

    # Boundary conditions at x=-4000
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([-4000], dtype=np.float64)), t_c).float() # x_c[0]
    bop3 = None
    # u(-4000,t)=1000
    bndval3 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval2.csv', header=None).values).reshape(-1)

    # Boundary conditions at x=4000
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([4000], dtype=np.float64)), t_c).float() # x_c[-1]
    bop4 = None
    # u(4000,t)=0
    bndval4 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval3.csv', header=None).values).reshape(-1)
    # Putting all bconds together,
    # bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], [bnd3, bop3, bndval3],[bnd4, bop4, bndval4]]
    bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]
    noise = False
    variance_arr = [0.001] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": data.ndim,
            "variance_arr": variance_arr
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "eq_search_iter": 100
        },
        "set_memory_properties": {
            "mem_for_cache_frac": 10
        },
        "set_moeadd_params": {
            "population_size": 15,  #
            "training_epochs": 5
        },
        "Cache_stored_tokens": {
            "token_type": "grid",
            "token_labels": ["t", "x"],
            "params_ranges": {"power": (1, 1)},
            "params_equality_ranges": None
        },
        "fit": {
            "max_deriv_order": (1, 1),
            "boundary": 20,  #
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": 2,
            "eq_sparsity_interval": (1e-8, 5.0),  #
            # "deriv_method": "ANN",  #
            # "deriv_method_kwargs": {"epochs_max": 1000},  #
            "deriv_method": "poly",
            "deriv_method_kwargs": {'smooth': True},
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "glob_epde": {
            "test_iter_limit": 100,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": True,
            "load_result": False
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 30,
            "lambda": 0.0001,
            "plot": False,
            "save_equations": True,
            "load_equations": False
        },
        "params": {
            "init_nodes": 'du/dx1{power: 1.0}_r'
        },
        "correct_structures": {
            "list_unique": ['du/dx1{power: 1.0}', 'du/dx2{power: 1.0} * u{power: 1.0}_r']
        }
    }

    solver_config = {
        "glob_solver": {
            "mode": "mat",
            "reverse": True
        },
        "Cache": {
            "use_cache": False,
            "save_always": False,
        },
        "Optimizer": {
            "learning_rate": 100,
            "lambda_bound": 5,
        },
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, param, bconds

# no changes
def KdV_equation():
    """

    """
    mesh = 100

    path = 'data/KdV_equation/'

    df = pd.read_csv(f'{path}KdV_solution/KdV_sln_{mesh}.csv', header=None)
    data = df.values
    data = np.transpose(data)  # x1 - t (axis Y), x2 - x (axis X)

    t_d = np.linspace(0, 1, mesh + 1)
    x_d = np.linspace(0, 1, mesh + 1)
    grid = np.meshgrid(t_d, x_d, indexing='ij')

    param = [x_d, t_d]

    noise = False
    variance_arr = [0.001] if noise else [0]

    bconds = False  # if there are no boundary conditions
    """
    Preparing boundary conditions (BC)

    For every boundary we define three items

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)

    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

    Meaning c1*u*d2u/dx2 has the form

    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}

    None is for function without derivatives

    bval=torch.Tensor prescribed values at every point in the boundary
    """

    x = torch.from_numpy(x_d)
    t = torch.from_numpy(t_d)

    # coefficients for BC

    a1, a2, a3 = [1, 2, 1]

    b1, b2, b3 = [2, 1, 3]

    r1, r2 = [5, 5]

    """
    Boundary x=0
    """

    # # points
    bnd1 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
    # operator a1*d2u/dx2+a2*du/dx+a3*u
    bop1 = {
        'a1*d2u/dx2':
            {
                'a1': a1,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            },
        'a2*du/dx':
            {
                'a2': a2,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            },
        'a3*u':
            {
                'a3': a3,
                'u': [None],
                'pow': 1,
                'var': 0
            }
    }

    # bop1=[[a1,[0,0],1],[a2,[0],1],[a3,[None],1]]

    # equal to zero
    bndval1 = torch.zeros(len(bnd1))

    """
    Boundary x=1
    """

    # points
    bnd2 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

    # operator b1*d2u/dx2+b2*du/dx+b3*u
    bop2 = {
        'b1*d2u/dx2':
            {
                'a1': b1,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            },
        'b2*du/dx':
            {
                'a2': b2,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            },
        'b3*u':
            {
                'a3': b3,
                'u': [None],
                'pow': 1,
                'var': 0
            }
    }

    # bop2=[[b1,[0,0],1],[b2,[0],1],[b3,[None],1]]
    # equal to zero
    bndval2 = torch.zeros(len(bnd2))

    """
    Another boundary x=1
    """
    # points
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

    # operator r1*du/dx+r2*u
    bop3 = {
        'r1*du/dx':
            {
                'r1': r1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            },
        'r2*u':
            {
                'r2': r2,
                'u': [None],
                'pow': 1,
                'var': 0
            }
    }

    # bop3=[[r1,[0],1],[r2,[None],1]]

    # equal to zero
    bndval3 = torch.zeros(len(bnd3))

    """
    Initial conditions at t=0
    """

    bnd4 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    # No operator applied,i.e. u(x,0)=0
    bop4 = None

    # equal to zero
    bndval4 = torch.zeros(len(bnd4))

    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], [bnd3, bop3, bndval3], [bnd4, bop4, bndval4]]

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "eq_search_iter": 100
        },
        "set_memory_properties": {
            "mem_for_cache_frac": 10
        },
        "set_moeadd_params": {
            "population_size": 20,  #
            "training_epochs": 8
        },
        "Cache_stored_tokens": {
            "token_type": "grid",
            "token_labels": ["t", "x"],
            "params_ranges": {"power": (1, 1)},
            "params_equality_ranges": None
        },
        "fit": {
            "max_deriv_order": (1, 3),
            "boundary": 15,  #
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": 2,
            "eq_sparsity_interval": (1e-8, 5.0),  #
            "deriv_method": "ANN",  #
            "deriv_method_kwargs": {"epochs_max": 10},  #
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
            "init_nodes": ['d^3u/dx2^3{power: 1.0}']
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

    with open(f'{path}ebs_config.json', 'w') as fp:
        json.dump(ebs_config, fp)

    cfg_ebs = config.Config(f'{path}ebs_config.json')

    cfg_ebs.set_parameter('Optimizer.optimizer', 'Adam')
    cfg_ebs.set_parameter('Optimizer.learning_rate', 1e-4)
    cfg_ebs.set_parameter('Optimizer.lambda_bound', 100)
    cfg_ebs.set_parameter('StopCriterion.eps', 1e-5)
    cfg_ebs.set_parameter('StopCriterion.no_improvement_patience', None)
    cfg_ebs.set_parameter('Cache.model_randomize_parameter', 1e-5)
    cfg_ebs.set_parameter('Verbose.verbose', True)
    cfg_ebs.set_parameter('NN.h', 0.01)

    return data, grid, cfg_ebs, param, bconds


def burgers_equation_small_grid():
    path = "data/burgers_equation_small_grid/"
    df = pd.read_csv(f'{path}burgers_sln_100.csv', header=None)
    data = df.values
    data = np.transpose(data)  # x1 - t (axis Y), x2 - x (axis X)

    derives = None

    x = np.linspace(-1000, 0, 101)
    t = np.linspace(0, 1, 101)
    grid = np.meshgrid(t, x, indexing='ij')

    param = [t, x]

    bconds = False  # if there are no boundary conditions

    noise = False
    variance_arr = [0.10] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": data.ndim,
            "variance_arr": variance_arr
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 10,  #
            "verbose_params": {"show_moeadd_epochs": True}
        },
        # "set_memory_properties": {
        #     "mem_for_cache_frac": 10
        # },
        "set_moeadd_params": {
            "population_size": 5,  #
            "training_epochs": 5
        },
        # "Cache_stored_tokens": {
        #     "token_type": "grid",
        #     "token_labels": ["t", "x"],
        #     "params_ranges": {"power": (1, 1)},
        #     "params_equality_ranges": None
        # },
        "fit": {
            "variable_names": ['u', ],
            "max_deriv_order": (1, 1),
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": {'factors_num': [1, 2], 'probas': [0.7, 0.3]},
            "eq_sparsity_interval": (1e-8, 5.0),  #
            # "deriv_method": "ANN",  #
            # "deriv_method_kwargs": {"epochs_max": 1000},  #
            "deriv_method": "poly",
            "deriv_method_kwargs": {'smooth': True},
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "results": {
            "level_num": 2
        },
        "glob_epde": {
            "test_iter_limit": 10,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": True,
            "load_result": False
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 35,
            "lambda": 0.00001,
            "plot": False,
            "save_equations": True,
            "load_equations": False
        },
        "params": {
            "init_nodes": 'du/dx1{power: 1.0}_r'
        },
        "correct_structures": {
            "list_unique": ['du/dx1{power: 1.0}_r', 'du/dx2{power: 1.0} * u{power: 1.0}']
        }
    }

    img_dir = f'{path}burgers_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": "mat",
            "reverse": False
        },
        "Cache": {
            "use_cache": False,
            "save_always": False,
        },
        "Optimizer": {
            "learning_rate": 100,
            "lambda_bound": 5,
        },
        "Plot": {
            "step_plot_print": False,
            "step_plot_save": True,
            "image_save_dir": img_dir,
        }
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, param, bconds


def hunter_prey():
    path = "data/hunter_prey/"

    t = np.load(f'{path}t.npy')
    data = np.load(f'{path}data.npy')
    x = data[:, 0]
    y = data[:, 1]
    data = [x, y]
    grid = [t, ]

    derives = None

    t = np.linspace(0, 8, 1000)
    param = [t, ]

    x0, y0 = 1., 1.
    bnd1_0 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    bndval1_0 = torch.from_numpy(np.array([[x0]], dtype=np.float64))
    bnd1_1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    bndval1_1 = torch.from_numpy(np.array([[y0]], dtype=np.float64))

    bconds = [[bnd1_0, bndval1_0, 0], [bnd1_1, bndval1_1, 1]]

    noise = False
    variance_arr = [0.10] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": x.ndim - 1,
            "variance_arr": variance_arr
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 10,  #
            "verbose_params": {"show_moeadd_epochs": False}
        },
        "set_moeadd_params": {
            "population_size": 5,  #
            "training_epochs": 100
        },
        "fit": {
            "variable_names": ['u', 'v'],  # list of objective function names
            "max_deriv_order": (1,),
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": {'factors_num': [1, 2], 'probas': [0.8, 0.2]}, # the amount of tokens in the term and their probability of occurrence
            "data_fun_pow": 1,  # the maximum degree of one token in the term
            "eq_sparsity_interval": (1e-10, 1e-2),  #
            "deriv_method": "poly",
            "deriv_method_kwargs": {'smooth': False},
        },
        "results": {
            "level_num": 1
        },
        "glob_epde": {
            "test_iter_limit": 50,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False,
            "load_result": True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 35,
            "lambda": 0.01,
        },
        "correct_structures": {
            "list_unique": ['v{power: 1.0} * u{power: 1.0}_v', 'v{power: 1.0}_v', 'dv/dx1{power: 1.0}_r_v',
                            'v{power: 1.0} * u{power: 1.0}_u', 'u{power: 1.0}_u', 'du/dx1{power: 1.0}_r_u']
        }
    }

    img_dir = f'{path}hunter_prey_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": "autograd",
            "reverse": False
        },
        "Cache": {
            "use_cache": True,
            "save_always": True,
            "cache_dir": f"{path}cache/",
            "model_randomize_parameter": 1e-5
        },
        "Optimizer": {
            "learning_rate": 1e-4,
            "lambda_bound": 100,
        },
        "NN": {
            "h": 0.00001
        },
        "StopCriterion": {
            "eps": 1e-6,
            "tmax": 5e6,
            "patience": 3,
            "no_improvement_patience": 500
        },
        "Plot": {
            "step_plot_print": False,
            "step_plot_save": False,
            "image_save_dir": img_dir,
        }
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, param, bconds
