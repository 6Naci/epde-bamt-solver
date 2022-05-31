import re
import numpy as np

regex = re.compile('freq:\s\d\S\d+')  # Using regular expression for frequency delete (sin/cos)


def dict_update(d_main, temp, coeff, k):
    arr_temp = temp.split(' * ')
    temp2 = f'{arr_temp[1]} * {arr_temp[0]}' if len(arr_temp) == 2 else ""

    if temp in d_main:
        d_main[temp] += [0 for i in range(k - len(d_main[temp]))] + [coeff]
    elif temp2 in d_main:  # case: if structure recorded b * a provided, that a * b already exists
        d_main[temp2] += [0 for i in range(k - len(d_main[temp2]))] + [coeff]
    else:
        d_main[temp] = [0 for i in range(k)] + [coeff]

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

    for soEq in res:  # soEq - объект класса 'epde.structure.SoEq'
        for equation in soEq:  # equation - объект класса 'epde.structure.Equation'
            equation_s = equation.structure[0].structure  # Список объектов класса 'epde.structure.Term'
            equation_c = equation.structure[0].weights_final  # Коэффициенты правой части
            text_form_eq = regex.sub('', equation.structure[0].text_form)  # Полная запись ур-ния
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

