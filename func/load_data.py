import numpy as np
import pandas as pd
import torch
import scipy.io
import json
from TEDEouS import config
from default_configs import DEFAULT_CONFIG_EBS

config.default_config = json.loads(DEFAULT_CONFIG_EBS)

global_modules = {
        "discovery_module": {
            "name_module": "SINDY"}  # "EPDE" or "SINDY"
}


def set_sindy_func():
    """
    задаем множители, участвующие в слагаемых итогового уравнения (sindy);
    ПРИМЕЧАНИЕ: аргументом лямбда-функций может быть только u (v)

        library_functions: список математических выражений
        library_functions_names: список строковых названий мат. выражений

    примеры написания лямбда-функций:
        library_functions = [lambda u: np.cos(u)*np.cos(u), lambda u: 1/u]
        library_functions_names = [lambda u: f'cos({u})sin({u})', lambda u: f'1/{u}']
    """

    library_functions = [lambda u: u, lambda u: u * u]  # эквиваленты def f1(u): return u; def f2(u): return u*u
    library_functions_names = [lambda u: u, lambda u: f'{u}{u}']  # принимают строковый тип ('u' или 'v')

    return [library_functions, library_functions_names]


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
            "deriv_method_kwargs": {"smooth": True},
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
            "sample_k": 35,
            "lambda": 0.0001,
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
            "mode": "NN"
        },
        "Cache": {
            "use_cache": True,
            "save_always": True,
        }
    }

    sindy_config = {
        "PDELibrary": {
            "derivative_order": 3,
            "include_bias": True,
            "is_uniform": True,
            "include_interaction": True
        },
        "set_optimizer": {
            "type": "SR3"
        },
        "STLSQ": {
            "threshold": 5,
            "alpha": 1e-5,
            "normalize_columns": True
        },
        "SR3": {
            "threshold": 7,
            "max_iter": 1000,
            "tol": 1e-15,
            "nu": 1e2,
            "thresholder": 'l0',
            "normalize_columns": True
        }
    }

    config_modules = {**global_modules,
                      **(epde_config if global_modules["discovery_module"]["name_module"] == "EPDE" else sindy_config),
                      **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, param, bconds


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
    t = mat['t'][0]
    x = mat['x'][0]

    dx = pd.read_csv(f'{path}wolfram_sln/burgers_sln_dx_256.csv', header=None)
    d_x = dx.values
    d_x = np.transpose(d_x)

    dt = pd.read_csv(f'{path}wolfram_sln/burgers_sln_dt_256.csv', header=None)
    d_t = dt.values
    d_t = np.transpose(d_t)

    derives = np.zeros(shape=(data.shape[0], data.shape[1], 2))
    derives[:, :, 0] = d_t
    derives[:, :, 1] = d_x

    # Create mesh
    grid = np.meshgrid(t, x, indexing='ij')

    param = [x, t]

    bconds = False  # if there are no boundary conditions

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
            "max_deriv_order": (1, 1),
            "boundary": 100,  #
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
            "test_iter_limit": 10,  # how many times to launch algorithm (one time - 2-3 equations)
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

    sindy_config = {
        "PDELibrary": {
            "derivative_order": 3,
            "include_bias": True,
            "is_uniform": True,
            "include_interaction": True
        },
        "set_optimizer": {
            "type": "SR3"
        },
        "STLSQ": {
            "threshold": 5,
            "alpha": 1e-5,
            "normalize_columns": True
        },
        "SR3": {
            "threshold": 7,
            "max_iter": 1000,
            "tol": 1e-15,
            "nu": 1e2,
            "thresholder": 'l0',
            "normalize_columns": True
        }
    }

    config_modules = {**global_modules,
                      **(epde_config if global_modules["discovery_module"]["name_module"] == "EPDE" else sindy_config),
                      **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, param, bconds


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
        },
        "Optimizer": {
            "learning_rate": 1e-4,
            "lambda_bound": 100,
            "optimizer": "Adam"
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
            "model_randomize_parameter": 1e-5
        },
        "NN": {
            "h": 0.01
        },
        "Verbose": {
            "verbose": True
        },
        "StopCriterion": {
            "eps": 1e-5,
            "no_improvement_patience": None
        }
    }

    sindy_config = {
        "PDELibrary": {
            "derivative_order": 3,
            "include_bias": True,
            "is_uniform": True,
            "include_interaction": True
        },
        "set_optimizer": {
            "type": "SR3"
        },
        "STLSQ": {
            "threshold": 5,
            "alpha": 1e-5,
            "normalize_columns": True
        },
        "SR3": {
            "threshold": 7,
            "max_iter": 1000,
            "tol": 1e-15,
            "nu": 1e2,
            "thresholder": 'l0',
            "normalize_columns": True
        }
    }

    config_modules = {**global_modules,
                      **(epde_config if global_modules["discovery_module"]["name_module"] == "EPDE" else sindy_config),
                      **bamt_config,
                      **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, cfg_ebs, param, bconds
