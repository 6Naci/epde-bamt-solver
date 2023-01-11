import math
import os
import pickle
import random

import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CacheStoredTokens, CustomTokens

from func import obj_collection as collection
from func.confidence_region import get_rms


def equation_fit(data, grid, derives, config_epde):
    dimensionality = config_epde.params["global_config"]["dimensionality"]

    deriv_method_kwargs = {}
    if config_epde.params["fit"]["deriv_method"] == "poly":
        deriv_method_kwargs = {'smooth': config_epde.params["fit"]["deriv_method_kwargs"]["smooth"], 'grid': grid}
    elif config_epde.params["fit"]["deriv_method"] == "ANN":
        deriv_method_kwargs = {'epochs_max': config_epde.params["fit"]["deriv_method_kwargs"]["epochs_max"]}

    epde_search_obj = epde_alg.epde_search(use_solver=config_epde.params["epde_search"]["use_solver"],
                                           dimensionality=dimensionality,
                                           boundary=config_epde.params["epde_search"]["boundary"],
                                           coordinate_tensors=grid,
                                           verbose_params=config_epde.params["epde_search"]["verbose_params"])

    # epde_search_obj.set_memory_properties(data, mem_for_cache_frac=config_epde.params["set_memory_properties"][
    #     "mem_for_cache_frac"])
    epde_search_obj.set_moeadd_params(population_size=config_epde.params["set_moeadd_params"]["population_size"],
                                      training_epochs=config_epde.params["set_moeadd_params"]["training_epochs"])

    # custom_grid_tokens = CacheStoredTokens(token_type=config_epde.params["Cache_stored_tokens"]["token_type"],
    #                                        boundary=config_epde.params["fit"]["boundary"],
    #                                        token_labels=config_epde.params["Cache_stored_tokens"]["token_labels"],
    #                                        token_tensors=dict(
    #                                            zip(config_epde.params["Cache_stored_tokens"]["token_labels"], grid)),
    #                                        params_ranges=config_epde.params["Cache_stored_tokens"]["params_ranges"],
    #                                        params_equality_ranges=config_epde.params["Cache_stored_tokens"][
    #                                            "params_equality_ranges"])
    '''
    Method epde_search.fit() is used to initiate the equation search.
    '''
    epde_search_obj.fit(data=data, variable_names=config_epde.params["fit"]["variable_names"],
                        data_fun_pow=config_epde.params["fit"]["data_fun_pow"],
                        max_deriv_order=config_epde.params["fit"]["max_deriv_order"],
                        equation_terms_max_number=config_epde.params["fit"]["equation_terms_max_number"],
                        equation_factors_max_number=config_epde.params["fit"]["equation_factors_max_number"],
                        coordinate_tensors=grid, eq_sparsity_interval=config_epde.params["fit"]["eq_sparsity_interval"],
                        derivs=[derives] if derives is not None else None,
                        deriv_method=config_epde.params["fit"]["deriv_method"],
                        deriv_method_kwargs=deriv_method_kwargs,
                        # additional_tokens=[custom_grid_tokens, ],
                        memory_for_cache=config_epde.params["fit"]["memory_for_cache"],
                        prune_domain=config_epde.params["fit"]["prune_domain"])

    '''
    The results of the equation search have the following format: if we call method 
    .equation_search_results() with "only_print = True", the Pareto frontiers 
    of equations of varying complexities will be shown, as in the following example:

    If the method is called with the "only_print = False", the algorithm will return list 
    of Pareto frontiers with the desired equations.
    '''

    epde_search_obj.equation_search_results(only_print=True, level_num=config_epde.params["results"]["level_num"])

    return epde_search_obj


def epde_equations(u, grid_u, derives, cfg, variance, title):
    # noise = []
    # for i in range(u.shape[0]):
    #     noise.append(np.random.normal(0, variance * get_rms(u[i, :]), u.shape[1]))  # for dimensionality == 2
    # noise = np.array(noise)
    #
    # u_total = u + noise
    # Creating results folder
    if not (os.path.exists(f'data/{title}/epde_result')):
        os.mkdir(f'data/{title}/epde_result')

    if cfg.params["glob_epde"]["load_result"]:
        # Need to check the existence of the file or send the path
        return pd.read_csv(f'data/{title}/epde_result/output_main_{title}.csv', index_col='Unnamed: 0', sep='\t', encoding='utf-8'), False

    k = 0  # number of equations (final)
    variable_names = cfg.params["fit"]["variable_names"] # list of objective function names
    table_main = [{i: [{}, {}]} for i in variable_names]  # dict/table coefficients left/right parts of the equation

    # Loading temporary data (for saving temp results)
    if os.path.exists(f'data/{title}/epde_result/table_main_general.pickle'):
        with open(f'data/{title}/epde_result/table_main_general.pickle', 'rb') as f:
            table_main = pickle.load(f)
        with open(f'data/{title}/epde_result/k_main_general.pickle', 'rb') as f:
            k = pickle.load(f)

    for test_idx in np.arange(cfg.params["glob_epde"]["test_iter_limit"]):
        epde_obj = equation_fit(u, grid_u, derives, cfg)
        res = epde_obj.equation_search_results(only_print=False, level_num=cfg.params["results"]["level_num"])  # result search

        table_main, k = collection.object_table(res, variable_names, table_main, k)
        # To save temporary data
        with open(f'data/{title}/epde_result/table_main_general.pickle', 'wb') as f:
            pickle.dump(table_main, f, pickle.HIGHEST_PROTOCOL)

        with open(f'data/{title}/epde_result/k_main_general.pickle', 'wb') as f:
            pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)

        print(test_idx)

    frame_main = collection.preprocessing_bamt(variable_names, table_main, k)

    if cfg.params["glob_epde"]["save_result"]:
        if os.path.exists(f'data/{title}/epde_result/output_main_{title}.csv'):
            frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}_{len(os.listdir(path=f"data/{title}/epde_result/"))}.csv', sep='\t', encoding='utf-8')
        else:
            frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}.csv', sep='\t', encoding='utf-8')

    return frame_main, epde_obj
