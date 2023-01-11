import os
import pandas as pd
import numpy as np
import torch
import sys
import time

import dill as pickle

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


def solver_equations(cfg, params, b_conds, equations, epde_obj=False, title=None):

    if not (os.path.exists(f'data/{title}/solver_result')):
        os.mkdir(f'data/{title}/solver_result')

    dim = cfg.params["global_config"]["dimensionality"] + 1  # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
    k_variable_names = len(cfg.params["fit"]["variable_names"])

    grid = grid_format_prepare(params, cfg.params["glob_solver"]["mode"]).to(device)
    set_solutions, models = [], []

    # for variant mode = "NN" and "autograd" (default) maybe is not best variant
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, k_variable_names)
    )

    for equation in equations:
        start = time.time()
        if not b_conds:
            text_form = view_for_create_eq(equation)
            eq_g = translate_equation(text_form, epde_obj.pool)  # generating object of the class 'epde.structure.Equation' on based search object from epde
            # eq_s = transform.solver_view(equation, config_solver, reverse=config_solver.params["glob_solver"]["reverse"]) # instead of a function solver_form()
            eq_s = eq_g.solver_form()
            if cfg.params["glob_solver"]["mode"] == 'mat':
                eq_s = coefficient_unification(grid, eq_s, mode=cfg.params["glob_solver"]["mode"])
            b_conds_g = eq_g.boundary_conditions(full_domain=True)

            principal_bcond_shape = b_conds_g[0][1].shape # if grid is only square, else !error!
            for i in range(len(b_conds_g)):
                b_conds_g[i][1] = b_conds_g[i][1].reshape(principal_bcond_shape)

            equation = Equation(grid, eq_s, b_conds_g, h=0.01).set_strategy(cfg.params["glob_solver"]["mode"])

        else:
            eq_s = transform.solver_view(equation, cfg)
            equation = Equation(grid, eq_s, b_conds, h=cfg.params["NN"]["h"]).set_strategy(cfg.params["glob_solver"]["mode"])

        if cfg.params["glob_solver"]["mode"] == 'mat':
            model = torch.rand(grid[0].shape)

            model_arch = torch.nn.Sequential(
                torch.nn.Linear(dim, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, k_variable_names)
            )

        model = Solver(grid, equation, model, cfg.params["glob_solver"]["mode"]).solve(
            lambda_bound=cfg.params['Optimizer']['lambda_bound'],
            cache_dir=cfg.params['Cache']['cache_dir'],
            cache_verbose=cfg.params['Cache']['cache_verbose'],
            save_always=cfg.params['Cache']['save_always'],
            use_cache=cfg.params['Cache']['use_cache'],
            model_randomize_parameter=cfg.params['Cache']['model_randomize_parameter'],
            verbose=cfg.params['Verbose']['verbose'],
            learning_rate=cfg.params['Optimizer']['learning_rate'],
            print_every=cfg.params['Verbose']['print_every'],
            no_improvement_patience=cfg.params['StopCriterion']['no_improvement_patience'],
            patience=cfg.params['StopCriterion']['patience'],
            eps=cfg.params['StopCriterion']['eps'], tmin=cfg.params['StopCriterion']['tmin'],
            tmax=cfg.params['StopCriterion']['tmax'],
            cache_model=None if cfg.params["glob_solver"]["mode"] != "mat" else model_arch,
            step_plot_print=cfg.params["Plot"]["step_plot_print"],
            step_plot_save=cfg.params["Plot"]["step_plot_save"],
            image_save_dir=cfg.params["Plot"]["image_save_dir"])

        end = time.time()
        print(f'Time = {end - start}')

        Solver(grid, equation, model, cfg.params["glob_solver"]["mode"]).solution_print(solution_print=True, solution_save=True, save_dir=cfg.params["Plot"]["image_save_dir"])
        model_main = Solver(grid, equation, model, cfg.params["glob_solver"]["mode"])
        models.append(model_main)

        solution_function = model if cfg.params["glob_solver"]["mode"] == "mat" else model(model_main.grid)
        solution_function = solution_function.reshape(len(params[0]), -1).detach().numpy() if dim > 1 else solution_function.detach().numpy()

        if not len(set_solutions):
            set_solutions = [solution_function]
        else:
            set_solutions.append(solution_function)
        # To save temporary solutions
        torch.save(np.array(set_solutions),
                   f'data/{title}/solver_result/file_u_main_{list(np.array(set_solutions).shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')

    set_solutions = np.array(set_solutions)

    number_of_files = int(len(os.listdir(path=f"data/{title}/solver_result/")))
    if os.path.exists(f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt'):
        torch.save(set_solutions, f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}_{number_of_files}.pt')
    else:
        torch.save(set_solutions, f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')

    # # Load data
    # set_solutions, models = torch.load(f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt'), None

    return set_solutions, models
