import pandas as pd
import numpy as np
import os
import re

import pysindy as ps

from func import eq_collection as collection

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# !!! Лучше потом в отдельную папке func перенести !!!
def sindy_out_format(model, max_deriv_order):
    def pure_derivs_format(string, max_deriv_order, deriv_symbol='\d'):
        # производная переменной x2 или x1
        if deriv_symbol == '\d':
            deriv_ind = 'x2'
        else:
            deriv_ind = 'x1'

        for i in range(max_deriv_order, 0, -1):

            derivs_str = deriv_symbol
            regex1 = r'x0_' + derivs_str * i + '[^1,t]'
            match1 = re.search(regex1, string)
            regex2 = r'x0_' + derivs_str * i + '$'
            match2 = re.search(regex2, string)

            # конец строки вынесен в отдельный паттерн regex2
            if match2 is not None:
                start, end = match2.regs[0][0], match2.regs[0][1]
                if i != 1:
                    string = string[:start] + 'd^' + str(i) + 'u/d' + deriv_ind + '^' + str(i) + '{power: 1}' \
                             + string[end:]
                else:
                    string = string[:start] + 'du/d' + deriv_ind + '{power: 1}' + string[end:]

            if match1 is None:
                continue
            else:
                start, end = match1.regs[0][0], match1.regs[0][1]
                if i == 1:
                    string = string[:start] + 'du/d' + deriv_ind + '{power: 1} ' + string[end:]
                else:
                    string = string[:start] + 'd^' + str(i) + 'u/d' + deriv_ind + '^' + str(i) + '{power: 1} ' \
                             + string[end:]
        return string

    def mix_derivs_format(string, match):
        if match is not None:
            start, end = match.regs[0][0], match.regs[0][1]

            # считаем число производных по x2 и x1 в найденном кусочке
            number_1 = string.count('1', start, end)
            number_t = i - number_1

            if number_1 == 1:
                string_x_der = ' du/dx2{power: 1}'
            else:
                string_x_der = ' d^' + str(number_1) + 'u/dx2^' + str(number_1) + '{power: 1}'

            if number_t == 1:
                string_t_der = '* du/dx1{power: 1} *'
            else:
                string_t_der = '* d^' + str(number_t) + 'u/dx1^' + str(number_t) + '{power: 1} *'

            string = string[:start] + string_t_der + string_x_der + string[end:]
        return string

    string_in = model.equations()[0]

    # заменяем нулевую степень
    string = string_in.replace(" 1 ", " ")

    # заменим все чистые производные
    string = pure_derivs_format(string, max_deriv_order)
    string = pure_derivs_format(string, max_deriv_order, 't')

    # заменим смешанные производные
    for i in range(max_deriv_order, 1, -1):
        derivs_str = '[1,t]'
        regex2 = r'x0_' + derivs_str * i + '$'
        match2 = re.search(regex2, string)
        string = mix_derivs_format(string, match2)

        regex1 = r'x0_' + derivs_str * i
        match1 = re.search(regex1, string)
        string = mix_derivs_format(string, match1)

    # проставим пробелы и * после числовых коэфф-в
    while True:
        regex = r'\d [a-zA-Z]'  # + '[^1]'
        match = re.search(regex, string)
        if match is None:
            break
        else:
            start = match.regs[0][0]
            string = string[:start + 1] + ' *' + string[start + 1:]

    # заменим x0 на u{power: 1} (нулевая степень производной)
    for j in range(10, 0, -1):
        asterixes = j - 1
        insert_string = 'u{power: 1} ' + '* u{power: 1} ' * asterixes
        string = string.replace('x0' * j, insert_string)

    # проставим недостающие * и пробелы
    while True:
        regex = r'power: 1} [a-zA-Z]'
        match = re.search(regex, string)
        if match is None:
            break
        else:
            start = match.regs[0][0]
            string = string[:start + 9] + ' *' + string[start + 9:]

    string_out = 'du/dx1 = ' + string
    return string_out


def equation_fit(data, grid, derives, config_sindy):
    """YOUR CODE HERE"""







def sindy_equations(u, grid_u, derives, cfg, variance, title):

    k = 0  # number of equations (final)
    dict_main, dict_right = {}, {}  # dict/table coeff the left/right part of the equation

    """YOUR CODE HERE"""

    np.random.seed(100)

    # Где этот параметр используется?
    integrator_keywords = {'rtol': 1e-12, 'method': 'LSODA', 'atol': 1e-12}

    '''
    df = pd.read_csv('KdV_sln_100.csv', header=None)
    u = df.values
    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])

    dt = t[1] - t[0]
    dx = x[1] - x[0]

    u = u.reshape(len(x), len(t), 1)
    
    Эти данные можно получить через load_data
    t и x через grid_u
    '''

    test_iter_limit = 10
    for test_idx in np.arange(test_iter_limit):
        # задаем свои токены через лямбда выражения
        library_functions = [lambda x: x, lambda x: x * x]  # , lambda x: np.cos(x)*np.cos(x)]#, lambda x: 1/x]
        library_function_names = [lambda x: x,
                                  lambda x: x + x]  # , lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']#, lambda x: '1/'+x]

        # ? проблема с использованием multiindices
        # multiindices=np.array([[0,1],[1,1],[2,0],[3,0]])

        pde_lib = ps.PDELibrary(library_functions=library_functions,
                                function_names=library_function_names,
                                derivative_order=3, spatial_grid=x,
                                # multiindices=multiindices,
                                implicit_terms=True, temporal_grid=t,
                                include_bias=True, is_uniform=True, include_interaction=True)
        feature_library = ps.feature_library.PolynomialLibrary(degree=3)

        optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,
                           thresholder='l0', normalize_columns=True)
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

        model.fit(u, t=dt)
        model.print()

        # второй оптимизатор
        # optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)
        # model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        # model.fit(u, t=dt)
        # model.print()

        # string = model.equations()  # вернет правую часть уравнения в виде списка строки
        # print(string)

        max_deriv_order = pde_lib.derivative_order  # вытаскиваем параметр derivative_order, чтобы передать дальше
        string1 = sindy_out_format(model, max_deriv_order)  # форматированная строка
        print(string1)

        dict_main, dict_right, k = """YOUR CODE HERE"""
        '''
        функция collection.eq_table(res, dict_main, dict_right, k) не подходит для sindy (нужно сделать по аналогии)
        '''
        print(test_idx)



    dict_main.update(dict_right)

    for key, value in dict_main.items():
        if len(value) < k:
            dict_main[key] = dict_main[key] + [0 for i in range(k - len(dict_main[key]))]


    frame = pd.DataFrame(dict_main)
    frame_main = frame.loc[:, (frame != 0).any(axis=0)]  # deleting empty columns

    # save
    if not (os.path.exists(f'data/{title}/sindy_result')):
        os.mkdir(f'data/{title}/sindy_result')

    if os.path.exists(f'data/{title}/sindy_result/output_main_{title}.csv'):
        frame_main.to_csv(
            f'data/{title}/epde_result/output_main_{title}_{len(os.listdir(path=f"data/{title}/sindy_result/"))}.csv',
            sep='\t', encoding='utf-8')
    else:
        frame_main.to_csv(f'data/{title}/sindy_result/output_main_{title}.csv', sep='\t', encoding='utf-8')

    # # Load data
    # frame_main = pd.read_csv(f'data/{title}/sindy_result/output_main_{title}.csv', index_col='Unnamed: 0', sep='\t', encoding='utf-8')

    return frame_main
