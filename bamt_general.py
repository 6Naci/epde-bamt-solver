import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bamt.Networks as Nets
import bamt.Nodes as Nodes
import bamt.Preprocessors as pp
from pgmpy.estimators import K2Score
import dill as pickle
from sklearn import preprocessing
import re
import itertools


def token_check(columns, objects_res, config_bamt):
    list_correct_structures_unique = config_bamt.params["correct_structures"]["list_unique"]
    variable_names = config_bamt.params["fit"]["variable_names"]

    list_correct_structures = set()
    for term in list_correct_structures_unique:
        str_r = '_r' if '_r' in term else ''
        str_elem = ''
        if any(f'_{elem}' in term for elem in variable_names):
            for elem in variable_names:
                if f'_{elem}' in term:
                    term = term.replace(f'_{elem}', "")
                    str_elem = f'_{elem}'
        # for case if several terms exist
        arr_term = re.sub('_r', '', term).split(' * ')
        perm_set = list(itertools.permutations([i for i in range(len(arr_term))]))
        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_r + str_elem
            list_correct_structures.add(temp)

    def out_red(text):
        print("\033[31m {}".format(text), end='')

    def out_green(text):
        print("\033[32m {}".format(text), end='')

    met, k_sys = 0, len(objects_res)
    k_min = k_sys if k_sys < 5 else 5

    for object_row in objects_res[:k_min]:
        k_c, k_l = 0, 0
        for col in columns:
            if col in object_row:
                if col in list_correct_structures:
                    k_c += 1
                    out_green(f'{col}')
                    print(f'\033[0m:{object_row[col]}')
                else:
                    k_l += 1
                    out_red(f'{col}')
                    print(f'\033[0m:{object_row[col]}')
        print(f'correct structures = {k_c}/{len(list_correct_structures_unique)}')
        print(f'incorrect structures = {k_l}')
        print('--------------------------')

    for object_row in objects_res:
        for temp in object_row.keys():
            if temp in list_correct_structures:
                met += 1

    print(f'average value (equation - {k_sys}) = {met / k_sys}')


def get_objects(synth_data, config_bamt):
    """
        Parameters
        ----------
        synth_data : pd.dataframe
            The fields in the table are structures of received systems/equations,
            where each record/row contains coefficients at each structure
        config_bamt:  class Config from TEDEouS/config.py contains the initial configuration of the task

        Returns
        -------
        objects_result - list objects (combination of equations or systems)
    """
    objects = []  # equations or systems
    for i in range(len(synth_data)):
        object_row = {}
        for col in synth_data.columns:
            object_row[synth_data[col].name] = synth_data[col].values[i]
        objects.append(object_row)

    objects_result = []
    for i in range(len(synth_data)):
        object_res = {}
        for key, value in objects[i].items():
            if abs(float(value)) > config_bamt.params["glob_bamt"]["lambda"]:
                object_res[key] = value
        objects_result.append(object_res)
        # print(f'{i + 1}.{object_res}')  # full string  output
        # print('--------------------------')

    return objects_result


def bs_experiment(df, cfg, title):

    if not (os.path.exists(f'data/{title}/bamt_result')):
        os.mkdir(f'data/{title}/bamt_result')

    if cfg.params["glob_bamt"]["load_result"]:
        with open(f'data/{title}/bamt_result/data_equations_{cfg.params["glob_bamt"]["sample_k"]}.pickle', 'rb') as f:
            return pickle.load(f)

    # Rounding values
    for col in df.columns:
        df[col] = df[col].round(decimals=10)
    # Deleting rows with condition
    df = df.loc[(df.sum(axis=1) != -len(cfg.params["fit"]["variable_names"])), (df.sum(axis=0) != 0)]
    # Deleting null columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # (df != 0).sum(axis = 0)

    df_initial = df.copy()

    for col in df.columns:
        if '_r' in col:
            df = df.astype({col: "int64"})
            df = df.astype({col: "str"})

    all_r = df.shape[0]
    unique_r = df.groupby(df.columns.tolist(), as_index=False).size().shape[0]

    print(f'Из {all_r} полученных систем \033[1m {unique_r} уникальных \033[0m ({int(unique_r / all_r * 100)} %)')

    l_r, l_left = [], []
    for term in list(df.columns):
        if '_r' in term:
            l_r.append(term)
        else:
            l_left.append(term)
    df = df[l_left + l_r]

    discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    encoder = preprocessing.LabelEncoder()
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    data, est = p.apply(df)
    info_r = p.info

    bn = Nets.HybridBN(has_logit=True, use_mixture=True)
    bn.add_nodes(info_r)

    df_temp = (df_initial[[col for col in df_initial.columns if '_r' in col]] != 0).copy()
    print(df_temp.sum(axis=0).sort_values(ascending=False)[:len(cfg.params["fit"]["variable_names"])])
    init_nodes_list = []
    for i in range(len(cfg.params["fit"]["variable_names"])):
        init_nodes = df_temp.sum(axis=0).idxmax()
        init_nodes_list.append(init_nodes)
        df_temp = df_temp.drop(init_nodes, axis=1)
    print(init_nodes_list)
    params = {"init_nodes": init_nodes_list} if not cfg.params["params"]["init_nodes"] else cfg.params[
        "params"]

    bn.add_edges(data, scoring_function=('K2', K2Score), params=params)
    bn.fit_parameters(df_initial)

    objects_res = []
    while len(objects_res) < cfg.params["glob_bamt"]["sample_k"]:
        synth_data = bn.sample(30, as_df=True)
        temp_res = get_objects(synth_data, cfg)

        if len(temp_res) + len(objects_res) > cfg.params["glob_bamt"]["sample_k"]:
            objects_res += temp_res[:cfg.params["glob_bamt"]["sample_k"] - len(objects_res)]
        else:
            objects_res += temp_res

    if cfg.params["correct_structures"]["list_unique"] is not None:
        token_check(df_initial.columns, objects_res, cfg)

    if cfg.params["glob_bamt"]["save_result"]:
        number_of_files = len(os.listdir(path=f"data/{title}/bamt_result/"))
        if os.path.exists(f'data/{title}/bamt_result/data_equations_{len(objects_res)}.csv'):
            with open(f'data/{title}/bamt_result/data_equations_{len(objects_res)}_{number_of_files}.pickle', 'wb') as f:
                pickle.dump(objects_res, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'data/{title}/bamt_result/data_equations_{len(objects_res)}.pickle', 'wb') as f:
                pickle.dump(objects_res, f, pickle.HIGHEST_PROTOCOL)

    return objects_res
