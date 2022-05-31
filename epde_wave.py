import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import Cache_stored_tokens

from func import eq_collection as collection


def epde_equation(df, test_iter_limit, mesh_wave, title):
    # df = pd.read_csv(f'data/{title}/wolfram_sln/wave_sln_{mesh_wave}.csv', header=None)

    k = 0  # number of equations (final)
    dict_main = {}  # dict/table coeff the left part of the equation
    dict_right = {}

    u = df.values
    u = np.transpose(u)

    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])

    boundary = 13
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    for test_idx in np.arange(test_iter_limit):
        epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter=100,
                                               dimensionality=dimensionality)

        epde_search_obj.set_memory_properties(u, mem_for_cache_frac=10)
        epde_search_obj.set_moeadd_params(population_size=10, training_epochs=5)

        custom_grid_tokens = Cache_stored_tokens(token_type='grid',
                                                 boundary=boundary,
                                                 token_labels=['t', 'x'],
                                                 token_tensors={'t': grids[0], 'x': grids[1]},
                                                 params_ranges={'power': (1, 1)},
                                                 params_equality_ranges=None)
        '''
        Method epde_search.fit() is used to initiate the equation search.
        '''

        epde_search_obj.fit(data=u, max_deriv_order=(2, 2), boundary=boundary,
                            equation_terms_max_number=3, equation_factors_max_number=1,
                            coordinate_tensors=grids, eq_sparsity_interval=(1e-8, 5.0),
                            deriv_method='poly', deriv_method_kwargs={'smooth': True, 'grid': grids},
                            additional_tokens=[custom_grid_tokens, ],
                            memory_for_cache=25, prune_domain=False)

        '''
        The results of the equation search have the following format: if we call method 
        .equation_search_results() with "only_print = True", the Pareto frontiers 
        of equations of varying complexities will be shown, as in the following example:
    
        If the method is called with the "only_print = False", the algorithm will return list 
        of Pareto frontiers with the desired equations.
        '''

        epde_search_obj.equation_search_results(only_print=True, level_num=2)

        res = epde_search_obj.equation_search_results(only_print=False, level_num=2)

        dict_main, dict_right, k = collection.eq_table(res, dict_main, dict_right, k)

        print(test_idx)

    dict_main.update(dict_right)

    for key, value in dict_main.items():
        if len(value) < k:
            dict_main[key] = dict_main[key] + [0 for i in range(k - len(dict_main[key]))]

    # Check result
    # for key, value in dict_main.items():
    #     print(f'{len(value)}: {value}')
    #     print(key)

    frame = pd.DataFrame(dict_main)
    frame_main = frame.loc[:, (frame != 0).any(axis=0)]  # deleting empty columns

    # print(frame_main)
    # frame_main.to_excel("output_wave.xlsx")
    # frame_main.to_csv(f'data/{title}/result/output_{mesh_wave}_main.csv', sep='\t', encoding='utf-8')

    return frame_main, epde_search_obj
