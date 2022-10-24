import os
import pandas as pd
import numpy as np
import torch
import sys
import time

import dill as pickle

from TEDEouS import solver
from func import transition_bs as transform

# from TEDEouS.input_preprocessing import grid_prepare
from TEDEouS.input_preprocessing import Equation
from TEDEouS.solver import Solver
from TEDEouS.solver import grid_format_prepare

from func.transition_bs import view_for_create_eq
from epde.interface.equation_translator import translate_equation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
device = torch.device('cpu')


def coefficient_unification(grid, operator, mode='NN'):
    for term in operator:
        coeff = term[0]
        if callable(coeff):
            coeff = coeff(grid)
            # coeff = coeff.reshape(-1, 1)
        elif type(coeff) == torch.Tensor:
            if mode == 'NN' or mode == 'autograd':
                if coeff.shape[0] != grid.shape[0]:
                    coeff = coeff.reshape(-1, 1)
                if coeff.shape[0] != grid.shape[0]:
                    print(
                        'ERROR: Coefficient shape {} is inconsistent with grid shape {}, it may be because coefficients were pre-computed for matrix grid'.format(
                            coeff.shape, grid.shape))
            elif mode == 'mat':
                if coeff.shape != grid.shape[1:]:
                    try:
                        coeff = coeff.reshape(grid.shape[1:]).T
                    except Exception:
                        print(
                            'ERROR: Something went wrong with the coefficient, it may be because coefficients were pre-computed for NN grid')
        term[0] = coeff
    return operator


def solver_equations(config_solver, params, b_conds, equations, epde_obj=False, title=None):

    grid = grid_format_prepare(params, config_solver.params["glob_solver"]["mode"]).to(device)
    u_main, models = [], []

    # for variant mode = "NN" and "autograd" (default) maybe is not best variant
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    for equation in equations:
        start = time.time()
        if not b_conds:
            text_form = view_for_create_eq(equation)
            eq_g = translate_equation(text_form, epde_obj.pool)  # generating object of the class 'epde.structure.Equation' on based search object from epde
            # eq_s = transform.solver_view(equation, config_solver, reverse=config_solver.params["glob_solver"]["reverse"]) # instead of a function solver_form()
            eq_s = eq_g.solver_form()
            if config_solver.params["glob_solver"]["mode"] == 'mat':
                eq_s = coefficient_unification(grid, eq_s, mode=config_solver.params["glob_solver"]["mode"])
            b_conds_g = eq_g.boundary_conditions(full_domain=True)

            principal_bcond_shape = b_conds_g[0][1].shape # if grid is only square, else !error!
            for i in range(len(b_conds_g)):
                b_conds_g[i][1] = b_conds_g[i][1].reshape(principal_bcond_shape)

            equation = Equation(grid, eq_s, b_conds_g, h=0.01).set_strategy(config_solver.params["glob_solver"]["mode"])

        else:
            eq_s = transform.solver_view(equation, config_solver, reverse=config_solver.params["glob_solver"]["reverse"])
            equation = Equation(grid, eq_s, b_conds, h=config_solver.params["NN"]["h"]).set_strategy(config_solver.params["glob_solver"]["mode"])

        if config_solver.params["glob_solver"]["mode"] == 'mat':
            model = torch.rand(grid[0].shape)

            model_arch = torch.nn.Sequential(
                torch.nn.Linear(2, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1)
            )

            # model_arch = torch.nn.Sequential(
            #     torch.nn.Linear(2, 256),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(256, 64),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(64, 1024),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(1024, 1)
            # )


        model = Solver(grid, equation, model, config_solver.params["glob_solver"]["mode"]).solve(
            lambda_bound=config_solver.params['Optimizer']['lambda_bound'],
            verbose=2, learning_rate=config_solver.params['Optimizer']['learning_rate'],
            eps=config_solver.params['StopCriterion']['eps'], tmin=config_solver.params['StopCriterion']['tmin'],
            tmax=config_solver.params['StopCriterion']['tmax'],
            cache_dir=config_solver.params['Cache']['cache_dir'],
            cache_verbose=False, save_always=False, use_cache=False,
            no_improvement_patience=config_solver.params['StopCriterion']['no_improvement_patience'],
            print_every=config_solver.params['Verbose']['print_every'],
            model_randomize_parameter=config_solver.params['Cache']['model_randomize_parameter'],
            cache_model=None if config_solver.params["glob_solver"]["mode"] != "mat" else model_arch,
            step_plot_print=config_solver.params["Plot"]["step_plot_print"],
            step_plot_save=config_solver.params["Plot"]["step_plot_save"],
            image_save_dir=config_solver.params["Plot"]["image_save_dir"])

        end = time.time()
        print(f'Time = {end - start}')

        Solver(grid, equation, model, config_solver.params["glob_solver"]["mode"]).solution_print()
        model_main = Solver(grid, equation, model, config_solver.params["glob_solver"]["mode"])
        models.append(model_main)

        u = model if config_solver.params["glob_solver"]["mode"] == "mat" else model(model_main.grid)
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

    number_of_files = int(len(os.listdir(path=f"data/{title}/solver_result/")))

    if os.path.exists(f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{config_solver.params["glob_solver"]["mode"]}_{config_solver.params["global_config"]["variance_arr"]}.pt'):
        torch.save(u_main, f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{config_solver.params["glob_solver"]["mode"]}_{config_solver.params["global_config"]["variance_arr"]}_{number_of_files}.pt')
    else:
        torch.save(u_main, f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{config_solver.params["glob_solver"]["mode"]}_{config_solver.params["global_config"]["variance_arr"]}.pt')

    # # Load data
    # u_main = torch.load(f'data/{title}/solver_result/file_u_main_{list(u_main.shape)}_{cfg.params["glob_solver"]["mode"]}.pt')

    return u_main, models
