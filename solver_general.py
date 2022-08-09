import os
import pandas as pd
import numpy as np
import torch
import sys
import time

import dill as pickle

from TEDEouS import solver
from func import transition_bs as transform

from TEDEouS.input_preprocessing import grid_prepare
from TEDEouS.solver import point_sort_shift_solver, grid_format_prepare

from func.transition_bs import view_for_create_eq
from epde.interface.equation_translator import translate_equation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append('../')
device = torch.device('cpu')


def solver_equations(config_solver, params, b_conds, equations, epde_obj=False, title=None):

    grid = grid_format_prepare(params, config_solver.params["glob_solver"]["mode"]).to(device)
    grid_main = grid if config_solver.params["glob_solver"]["mode"] == "mat" else grid_prepare(grid)[0]  # prepared_grid
    u_main, models = [], []

    for equation in equations:
        if not b_conds:
            text_form = view_for_create_eq(equation)
            eq_g = translate_equation(text_form, epde_obj.pool)  # generating object of the class 'epde.structure.Equation' on based search object from epde
            eq_s = eq_g.solver_form()
            b_conds = eq_g.boundary_conditions(full_domain=True)

            principal_bcond_shape = b_conds[0][1].shape # if grid is only square, else !error!
            for i in range(len(b_conds)):
                b_conds[i][1] = b_conds[i][1].reshape(principal_bcond_shape)
        else:
            eq_s = transform.solver_view(equation, config_solver)

        start = time.time()
        model = solver.optimization_solver(params, None, eq_s, b_conds, config_solver, mode=config_solver.params["glob_solver"]["mode"])
        end = time.time()
        print(f'Time = {end - start}')

        models.append(model)
        u = model if config_solver.params["glob_solver"]["mode"] == "mat" else model(grid_main)
        u = u.reshape(len(params[0]), -1)
        u = u.detach().numpy()

        if not len(u_main):
            u_main = [u]
        else:
            u_main.append(u)

    u_main = np.array(u_main)

    # save solution
    if not (os.path.exists(f'data/{title}/solver_result')):
        os.mkdir(f'data/{title}/solver_result')

    number_of_files = int(len(os.listdir(path=f"data/{title}/solver_result/")) // 2)

    if os.path.exists(f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{config_solver.params["glob_solver"]["mode"]}.pt'):
        torch.save(u_main, f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{config_solver.params["glob_solver"]["mode"]}_{number_of_files}.pt')
    else:
        torch.save(u_main, f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{config_solver.params["glob_solver"]["mode"]}.pt')

    if os.path.exists(f'data/{title}/solver_result/file_prepared_grid_main_{list(grid_main.size())}_{config_solver.params["glob_solver"]["mode"]}.pt'):
        torch.save(grid_main, f'data/{title}/solver_result/file_prepared_grid_main_{list(grid_main.size())}_{config_solver.params["glob_solver"]["mode"]}_{number_of_files}.pt')
    else:
        torch.save(grid_main, f'data/{title}/solver_result/file_prepared_grid_main_{list(grid_main.size())}_{config_solver.params["glob_solver"]["mode"]}.pt')

    # # Load data
    # u_main = torch.load(f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{cfg.params["glob_solver"]["mode"]}.pt')
    # grid_main = torch.load(f'data/{title}/solver_result/file_prepared_grid_main_{list(grid_main.size())}_{cfg.params["glob_solver"]["mode"]}.pt')

    return u_main, grid_main
