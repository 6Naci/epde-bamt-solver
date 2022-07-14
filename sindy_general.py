import pandas as pd
import os

import numpy as np
import re
import itertools

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







def sindy_equations(u, x, t, cfg, variance, title):

    k = 0  # number of equations (final)
    dict_main, dict_right = {}, {}  # dict/table coeff the left/right part of the equation

    """YOUR CODE HERE"""

    np.random.seed(100)

    df = pd.read_csv('KdV_sln_100.csv', header=None)
    u = df.values
    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])

    dt = t[1] - t[0]
    u = u.reshape(len(x), len(t), 1)
    
    # Эти данные можно получить через load_data
    # t и x через grid_u

    test_iter_limit = 1
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
                                # implicit_terms=True, temporal_grid=t,
                                include_bias=True, is_uniform=True, include_interaction=True)
        # feature_library = ps.feature_library.PolynomialLibrary(degree=3)

        optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,
                           thresholder='l0', normalize_columns=True)
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

        model.fit(u, t=dt)

        # второй оптимизатор
        # optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)
        # model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        # model.fit(u, t=dt)

        # string = model.equations()  # вернет правую часть уравнения в виде списка строки
        # print(string)

        max_deriv_order = pde_lib.derivative_order  # вытаскиваем параметр derivative_order, чтобы передать дальше
        string1 = sindy_out_format(model, max_deriv_order)  # форматированная строка
        # print(string1)

        dict_main, dict_right, k = eq_table(model, dict_main, dict_right, k)
        # collection.eq_table()
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


def dict_update(d_main, term, coeff, k):

    str_t = '_r' if '_r' in term else ''
    arr_term = re.sub('_r', '', term).split(' * ')

    # if structure recorded b * a provided, that a * b already exists (for all case - generalization)
    perm_set = list(itertools.permutations([i for i in range(len(arr_term))]))
    structure_added = False

    for p_i in perm_set:
        temp = " * ".join([arr_term[i] for i in p_i]) + str_t
        if temp in d_main:
            d_main[temp] += [0 for i in range(k - len(d_main[temp]))] + [coeff]
            structure_added = True

    if not structure_added:
        d_main[term] = [0 for i in range(k)] + [coeff]

    return d_main


def eq_table(model, dict_main, dict_right, k):
    """
    Сбор полученных ур-ний (коэффициентов и структур) в общую таблицу (рассматривается отдельно правая и левая часть ур-ния)

    Parameters
        ----------
        res : Фронт Парето обнаруженных ур-ний
        k : Кол-во ур-ний (итоговое)
        dict_main : Словарь/таблица коэффициентов левой части ур-ий
        dict_right : -//- правой части ур-ний

        Returns
        -------
        dict_main, dict_right, k
    """

    features_names_model = model.get_feature_names()
    equation_c = model.coefficients()[0]

    text_form_eq, proc_features = format_terms_text(features_names_model, equation_c)# full equation line

    flag = False  # Флаг для правой части

    for term in proc_features:
        for t in range(len(equation_c)):
            c = equation_c[t] # коэффициент для соответствующего term
            if f'{c} * {term} +' in text_form_eq:
                dict_main = dict_update(dict_main, term, c, k)
                equation_c = np.delete(equation_c, t)
                break
            elif f'+ {c} =' in text_form_eq:
                dict_main = dict_update(dict_main, "C", c, k)
                equation_c = np.delete(equation_c, t)
                break
        sdf = f'= {term}'
        jki = text_form_eq[text_form_eq.find('='):]
        boool = sdf==jki
        if f'= {term}' == text_form_eq[text_form_eq.find('='):] and flag is False:
            flag = True
            term += '_r'
            dict_right = dict_update(dict_right, term, -1, k)
    k += 1

    print(k)

    return dict_main, dict_right, k


def format_terms_text(features, coefs):
    ls_process_terms = []
    ls = []
    if features[0] == '1' and coefs[0] != 0.:
        ls.append(str(coefs[0]))

    for i in range(1, len(features)):
        if coefs[i] != 0.:
            ls_process_terms.append(process_term(features[i]))
            ls.append(str(coefs[i]) + ' * ' + ls_process_terms[-1])

    text = ' + '.join(ls) + ' = du/dx1{power: 1.0}'
    ls_process_terms.append('du/dx1{power: 1.0}')
    return text, ls_process_terms


def process_term(term):

    def deriv_format(count, type='x1'):
        new_term = ''
        if count != 0:
            if count == 1:
                new_term= 'du/d' + type + '{power: 1.0}'
            else:
                new_term = 'd^' + str(count) + 'u/d' + type + '^' + str(count) + '{power: 1.0}'
        return new_term

    ind = term.find('x0_')
    new_termt = ''
    new_termx = ''
    if ind != -1:
        count_derivs = len(term) - ind - 3
        count_1 = term.count('1', ind+2)
        count_t = count_derivs-count_1
        new_termt = deriv_format(count_t)
        new_termx = deriv_format(count_1, 'x2')

    count_x0 = term.count('x0') - int('x0_' in term)
    ls_of_u = ['u{power: 1.0}'] * count_x0
    new_termu = ' * '.join(ls_of_u)

    ls_all = [new_termu, new_termt, new_termx]
    ls_all = list(filter(None, ls_all))
    new_term = ' * '.join(ls_all)
    return new_term


sindy_equations(1, 1, 1, 1, 1, 'sindy_test')