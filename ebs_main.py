import pandas as pd
import numpy as np
import torch
from epde.interface.equation_translator import translate_equation
import epde.globals as global_var

from epde_wave import epde_equation
import bamt_wave as bamt
import solver_wave as solver_eq

from TEDEouS import solver
from func import confidence_region as conf_plt
from func.transition_bs import view_for_create_eq
from func import transition_bs as transform
import dill as pickle


if __name__ == '__main__':

    # initial params before fit-EPDE (global params)
    grid_res = 70
    title = 'wave_equation' # name of the problem/equation
    test_iter_limit = 1
    # Load data
    df = pd.read_csv(f'data/{title}/wolfram_sln/wave_sln_{grid_res}.csv', header=None)

    df_main, epde_search_obj = epde_equation(df, test_iter_limit, grid_res, title)
    # need dimensionality, max_deriv_order on exit

    sample_k = 2
    equations = bamt.bs_experiment(df_main, sample_k, grid_res, title)

    # # save and open object
    # with open(f'data/{title}/data_object.pickle', 'wb') as f:
    #     pickle.dump(epde_search_obj, f, pickle.HIGHEST_PROTOCOL)

    # with open(f'data/{title}/data_object.pickle', 'rb') as f:
    #     epde_search_obj = pickle.load(f)
    #
    # with open(f'data/{title}/data_cache.pickle', 'rb') as f:
    #     cache = pickle.load(f)

    # need to save the equations through pickle

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

        exp_dict_list_main.append(exp_dict_list)
        exp_dict_list_flatten = [item for sublist in exp_dict_list_main for item in sublist]
        df = pd.DataFrame(exp_dict_list_flatten)
        df.to_csv('data/wave_equation/cache/wave_experiment_matrix_ebs_2_{}_cache={}.csv'.format(grid_res, str(CACHE)))

    u_main = np.array(u_main)
    #
    # # save solution
    # torch.save(u_main, f'data/{title}/solution/file_u_main_matrix.pt')
    # torch.save(prepared_grid_main, f'data/{title}/solution/file_prepared_grid_main_matrix.pt')


    # for equation in equations:
    #     u, prepared_grid, exp_dict_list = solver_eq.solver_equation(equation, grid_res, CACHE, title)
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
    # df = pd.DataFrame(exp_dict_list_flatten)
    # df.to_csv(f'data/{title}/cache/{title}_ebs_2_{grid_res}_cache={str(CACHE)}.csv')
    # u_main = np.array(u_main)

    # # save solution
    # torch.save(u_main, f'data/{title}/solution/file_u_main.pt')
    # torch.save(prepared_grid_main, f'data/{title}/solution/file_prepared_grid_main.pt')

    # # load solution
    # u_main = torch.load(f'data/{title}/solution/file_u_main.pt')
    # prepared_grid_main = torch.load(f'data/{title}/solution/file_prepared_grid_main.pt')

    # conf_plt.confidence_region_print(u_main, prepared_grid_main)
