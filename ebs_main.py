import pandas as pd
import numpy as np
import torch
from epde.interface.equation_translator import translate_equation
import epde.globals as global_var

from epde_wave import epde_equation
import bamt_wave as bamt
import solver_wave as solver_eq

from TEDEouS import solver
from func import load_data
from func import confidence_region as conf_plt
from func.transition_bs import view_for_create_eq
from func import transition_bs as transform
import dill as pickle

if __name__ == '__main__':

    tasks = {
        'wave_equation': load_data.wave_equation,
        'burgers_equation': load_data.burgers_equation
    }
    # initial params before fit-EPDE (global params)
    title = list(tasks.keys())[0]  # name of the problem/equation

    grid_res = 60

    u, grid_u = tasks[title](grid_res)

    test_iter_limit = 3  # how many times to launch algorithm (one time - 2-3 equations)
    noise = False
    variance_arr = [0.001] if noise else [0]

    for variance in variance_arr:
        df_main, epde_search_obj = epde_equation(u, test_iter_limit, grid_res, title, variance)
        # need dimensionality, max_deriv_order on exit

        # # Load data and Preprocessing
        # df_main = pd.read_csv(f'data/{title}/result/output_{grid_res}_main.csv', index_col='Unnamed: 0', sep='\t',
        #                       encoding='utf-8')

        sample_k = 2
        equations = bamt.bs_experiment(df_main, sample_k, grid_res, title)

        # with open(f'data/{title}/solution/data_equations_{sample_k}_var_{str(variance)}.pickle', 'wb') as f:
        #     pickle.dump(equations, f, pickle.HIGHEST_PROTOCOL)

        # with open(f'data/{title}/solution/data_equations_{sample_k}_var_{str(variance)}.pickle', 'rb') as f:
        #     equations = pickle.load(f)

        exp_dict_list_main = []
        u_main, prepared_grid_main = [], []
        CACHE = True

        # for solver.matrix_optimizer
        solver_inp = []
        for equation in equations:
            text_form = view_for_create_eq(equation)
            eq = translate_equation(text_form, epde_search_obj.pool)

            equation_main = transform.solver_view(equation)
            # eq = translate_equation(text_form, pool)
            solver_inp.append((eq.solver_form(), eq.boundary_conditions()))
            u, prepared_grid, exp_dict_list = solver_eq.solver_equation_matrix(grid_res, CACHE, eq, equation_main, title)
            u = u.reshape(-1, grid_res + 1)
            u = u.detach().numpy()
            if not len(u_main):
                u_main = [u]
                prepared_grid_main = prepared_grid
            else:
                u_main.append(u)

        # exp_dict_list_main.append(exp_dict_list)
        # exp_dict_list_flatten = [item for sublist in exp_dict_list_main for item in sublist]
        # df = pd.DataFrame(exp_dict_list_flatten)
        # df.to_csv('data/wave_equation/cache/wave_experiment_matrix_ebs_2_{}_cache={}.csv'.format(grid_res, str(CACHE)))
        #
        u_main = np.array(u_main)
        #
        # # save solution
        # torch.save(u_main, f'data/{title}/solution/file_u_main_matrix.pt')
        # torch.save(prepared_grid_main, f'data/{title}/solution/file_prepared_grid_main_matrix.pt')

        # for equation in equations:
        #     u, prepared_grid, exp_dict_list, model = solver_eq.solver_equation(equation, grid_res, CACHE, title)
        #     u = u.reshape(-1, grid_res + 1)
        #     u = u.detach().numpy()
        #     if not len(u_main):
        #         u_main = [u]
        #         prepared_grid_main = prepared_grid
        #     else:
        #         u_main.append(u)
        #
        #     exp_dict_list_main.append(exp_dict_list)
        #
        # exp_dict_list_flatten = [item for sublist in exp_dict_list_main for item in sublist]
        # df_res = pd.DataFrame(exp_dict_list_flatten)
        # df_res.to_csv(f'data/{title}/cache/{title}_ebs_2_{grid_res}_cache={str(CACHE)}.csv')
        # u_main = np.array(u_main)
        #
        # # save solution
        # torch.save(u_main, f'data/{title}/solution/file_u_main_{sample_k}_var_{str(variance)}.pt')
        # torch.save(prepared_grid_main, f'data/{title}/solution/file_prepared_grid_main_var_{str(variance)}.pt')

        # # load solution
        # u_main = torch.load(f'data/{title}/solution/file_u_main_{sample_k}_var_{str(variance)}.pt')
        # prepared_grid_main = torch.load(f'data/{title}/solution/file_prepared_grid_main_var_{str(variance)}.pt')

        conf_plt.confidence_region_print(u, grid_u, u_main, prepared_grid_main, variance)
