import os
import pandas as pd
import numpy as np
import torch
import sys
import time

from TEDEouS import solver

from func import transition_bs as transform

from TEDEouS.input_preprocessing import grid_prepare, bnd_prepare, operator_prepare
from TEDEouS.metrics import point_sort_shift_loss
from TEDEouS.solver import point_sort_shift_solver


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
device = torch.device('cpu')


def solver_equation_matrix(grid_res, CACHE, equation, equation_main, title):
    exp_dict_list = []

    x_grid = np.linspace(0, 1, grid_res + 1)
    t_grid = np.linspace(0, 1, grid_res + 1)

    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)

    # grid = []
    # grid.append(x)
    # grid.append(t)
    #
    # grid = np.meshgrid(*grid)
    # grid = torch.tensor(grid, device=device)

    coord_list = [x, t]

    grid = solver.grid_format_prepare(coord_list, mode='mat')
    grid.to(device)

    sln = np.genfromtxt(f'data/{title}/wolfram_sln/wave_sln_{grid_res}.csv', delimiter=',')
    sln_torch = torch.from_numpy(sln)
    sln_torch1 = sln_torch.reshape(-1, 1)  # I don't know a few lines, anyway, 1 column

    model_arch = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    lp_par = {'operator_p': 2,
              'operator_weighted': False,
              'operator_normalized': False,
              'boundary_p': 1,
              'boundary_weighted': False,
              'boundary_normalized': False}

    start = time.time()

    matrix_model = solver.matrix_optimizer(grid, None, equation.solver_form(), equation.boundary_conditions(), lambda_bound=10,
                                           verbose=True, learning_rate=1e-3, eps=1e-5, tmin=1000, tmax=1e6,
                                           use_cache=CACHE, cache_dir='../cache/', cache_verbose=True,
                                           batch_size=None, save_always=True,  lp_par=lp_par, print_every=None,
                                           patience=5, loss_oscillation_window=100,
                                           no_improvement_patience=1000,
                                           model_randomize_parameter=1e-6, optimizer='Adam',
                                           cache_model=model_arch)

    end = time.time()

    error_rmse = torch.sqrt(torch.mean((sln_torch1 - matrix_model(grid)) ** 2))

    prepared_grid, grid_dict, point_type = grid_prepare(grid)

    u = matrix_model(prepared_grid)

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().numpy(),
                          'type': 'wave_eqn', 'cache': CACHE})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return u, prepared_grid, exp_dict_list


def solver_equation(equation, grid_res, CACHE, title):
    exp_dict_list = []

    x_grid = np.linspace(0, 1, grid_res + 1)
    t_grid = np.linspace(0, 1, grid_res + 1)

    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)

    grid = torch.cartesian_prod(x, t).float()  # Do cartesian product of the given sequence of tensors.

    grid.to(device)

    """
    Preparing boundary conditions (BC)

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bval=torch.Tensor prescribed values at every point in the boundary

    """

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    bop1 = None

    # u(0,x)= 10000*sin[1/10 x*(x - 1)]^2
    bndval1 = 10000 * torch.sin((0.1 * bnd1[:, 0] * (bnd1[:, 0] - 1)) ** 2)

    # Initial conditions at t=0
    bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()

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
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()

    bop3 = None

    # u(0,t)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()

    bop4 = None
    # u(1,t)=0
    bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64))

    # Putting all bconds together
    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], [bnd3, bop3, bndval3], [bnd4, bop4, bndval4]]

    """
    Defining wave equation

    Operator has the form
    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0
    Function exist solver_view from func import transition_bs as transform

    """

    equation_main = transform.solver_view(equation)

    sln = np.genfromtxt(f'data/{title}/wolfram_sln/wave_sln_{grid_res}.csv', delimiter=',')
    sln_torch = torch.from_numpy(sln)
    sln_torch1 = sln_torch.reshape(-1, 1)  # I don't know a few lines, anyway, 1 column

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    lp_par = {'operator_p': 2,
              'operator_weighted': False,
              'operator_normalized': False,
              'boundary_p': 1,
              'boundary_weighted': False,
              'boundary_normalized': False}

    start = time.time()

    model = point_sort_shift_solver(grid, model, equation_main, bconds, lambda_bound=10, verbose=2, learning_rate=1e-3,
                                    h=abs((t[1] - t[0]).item()),
                                    eps=1e-5, tmin=1000, tmax=1e6, use_cache=CACHE, cache_dir='../cache/',
                                    cache_verbose=True
                                    , batch_size=None, save_always=True, lp_par=lp_par, print_every=None,
                                    model_randomize_parameter=1e-6)
    end = time.time()


    error_rmse = torch.sqrt(torch.mean((sln_torch1 - model(grid)) ** 2))

    prepared_grid, grid_dict, point_type = grid_prepare(grid)

    u = model(prepared_grid)

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().numpy(),
                          'type': 'wave_eqn', 'cache': CACHE})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    return u, prepared_grid, exp_dict_list

# n_runs = 1
# grid_res = 70
#
# exp_dict_list_main = []
#
# CACHE = True
#
# # view after BAMT (one/list equation/s)
# equation = {'d^2u/dx2^2{power: 1.0}': 0.02036869782557119, 'u{power: 1.0}': -0.6043591746687335,
#             'C': 0.9219325066472699, 'd^2u/dx1^2{power: 1.0}_r': '-1'}
#
# # for grid_res in range(10, 101, 10):
# for _ in range(n_runs):
#     # for equation in equations:
#     exp_dict_list_main.append(wave_experiment(grid_res, CACHE, equation))
#     exp_dict_list_flatten = [item for sublist in exp_dict_list_main for item in sublist]
#     df = pd.DataFrame(exp_dict_list_flatten)
#     # df.boxplot(by='grid_res', column='time', fontsize=42, figsize=(20, 10))
#     # df.boxplot(by='grid_res', column='RMSE', fontsize=42, figsize=(20, 10), showfliers=False)
#     # df.to_csv('data/wave_equation/cache/wave_experiment_2_{}_cache={}.csv'.format(grid_res, str(CACHE)))
