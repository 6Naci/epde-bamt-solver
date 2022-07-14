import pandas as pd
import os
import numpy as np
import pysindy as ps
from func import eq_collection as collection
# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def equation_fit(data, grid, config_sindy):
    t = grid[0][:, 0]
    x = grid[1][0, :]
    dt = t[1] - t[0]

    data = np.transpose(data)
    data = data.reshape(len(x), len(t), 1)

    library_functions = [lambda x: x, lambda x: x * x]  # , lambda x: np.cos(x)*np.cos(x)]#, lambda x: 1/x]
    library_function_names = [lambda x: x,
                              lambda x: x + x]  # , lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']#, lambda x: '1/'+x]

    # ? проблема с использованием multiindices
    # multiindices=np.array([[0,1],[1,1],[2,0],[3,0]])

    pde_lib = ps.PDELibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            derivative_order=config_sindy.params["PDELibrary"]["derivative_order"],
                            spatial_grid=x,
                            # multiindices=multiindices,
                            # implicit_terms=True, temporal_grid=t,
                            include_bias=config_sindy.params["PDELibrary"]["include_bias"],
                            is_uniform=config_sindy.params["PDELibrary"]["is_uniform"],
                            include_interaction=config_sindy.params["PDELibrary"]["include_interaction"])


    # feature_library = ps.feature_library.PolynomialLibrary(degree=3)

    optimizer = ps.SR3(threshold=config_sindy.params["SR3"]["threshold"],
                       max_iter=config_sindy.params["SR3"]["max_iter"],
                       tol=config_sindy.params["SR3"]["tol"],
                       nu=config_sindy.params["SR3"]["nu"],
                       thresholder=config_sindy.params["SR3"]["thresholder"],
                       normalize_columns=config_sindy.params["SR3"]["normalize_columns"])

    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(data, t=dt)

    # второй оптимизатор
    # optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)
    # model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    # model.fit(u, t=dt)

    return model


def sindy_equations(u, grid, cfg, variance, title):

    k = 0  # number of equations (final)
    dict_main, dict_right = {}, {}  # dict/table coeff the left/right part of the equation

    np.random.seed(100)

    test_iter_limit = 1
    for test_idx in np.arange(test_iter_limit):
        model = equation_fit(u, grid, cfg)
        dict_main, dict_right, k = collection.eq_table_sindy(model, dict_main, dict_right, k)
        print(test_idx)

    dict_main.update(dict_right)
    for key, value in dict_main.items():
        if len(value) < k:
            dict_main[key] = dict_main[key] + [0 for i in range(k - len(dict_main[key]))]

    frame = pd.DataFrame(dict_main)
    frame_main = frame.loc[:, (frame != 0).any(axis=0)]  # deleting empty columns

    # save
    if not (os.path.exists(f'data/{title}')):
        os.mkdir(f'data/{title}')

    if not (os.path.exists(f'data/{title}/sindy_result')):
        os.mkdir(f'data/{title}/sindy_result')

    if os.path.exists(f'data/{title}/sindy_result/output_main_{title}.csv'):
        frame_main.to_csv(
            f'data/{title}/sindy_result/output_main_{title}_{len(os.listdir(path=f"data/{title}/sindy_result/"))}.csv',
            sep='\t', encoding='utf-8')
    else:
        frame_main.to_csv(f'data/{title}/sindy_result/output_main_{title}.csv', sep='\t', encoding='utf-8')

    # # Load data
    # frame_main = pd.read_csv(f'data/{title}/sindy_result/output_main_{title}.csv', index_col='Unnamed: 0', sep='\t', encoding='utf-8')

    return frame_main, model
