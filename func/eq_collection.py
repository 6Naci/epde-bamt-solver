import re
import numpy as np
import itertools

regex = re.compile('freq:\s\d\S\d+')  # Using regular expression for frequency delete (sin/cos)


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


def eq_table(res, dict_main, dict_right, k):
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

    for soEq in res:  # soEq - an object of the class 'epde.structure.SoEq'
        for equation in soEq:  # equation - an object of the class 'epde.structure.Equation'
            equation_s = equation.structure[0].structure  # list of class objects 'epde.structure.Term'
            equation_c = equation.structure[0].weights_final  # coefficients of the right part
            text_form_eq = regex.sub('', equation.structure[0].text_form)  # full equation line
            # equation.structure[0].solver_form() - тензорное представление ур-ния
            # epde.factor.Factor (deriv_code[0] - производная токена, equality_ranges - степень/размерность параметра,
            #                                     cache_label - название токена)
            flag = False  # Флаг для правой части
            for t_eq in equation_s:
                term = regex.sub('', t_eq.name)  # Полное имя слагаемого
                for t in range(len(equation_c)):
                    c = equation_c[t]
                    if f'{c} * {term} +' in text_form_eq:
                        dict_main = dict_update(dict_main, term, c, k)
                        equation_c = np.delete(equation_c, t)
                        break
                    elif f'+ {c} =' in text_form_eq:
                        dict_main = dict_update(dict_main, "C", c, k)
                        equation_c = np.delete(equation_c, t)
                        break
                if f'= {term}' == text_form_eq[text_form_eq.find('='):] and flag is False:
                    flag = True
                    term += '_r'
                    dict_right = dict_update(dict_right, term, -1, k)
            k += 1

    print(k)

    return dict_main, dict_right, k
