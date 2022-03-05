import numpy as np
import matplotlib.pyplot as plt
import tabulate


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def find_percentile(cdf_dict, percentile):
    sorted_dict_tuples = sorted(cdf_dict.items())
    cdf_indices = [sorted_dict_tuples[i][0] for i in range(len(sorted_dict_tuples))]
    cdf_values = [sorted_dict_tuples[i][1] for i in range(len(sorted_dict_tuples))]
    if percentile >= cdf_values[-1]:
        return cdf_indices[-1]
    if percentile <= cdf_values[0]:
        return cdf_indices[0]

    high = len(cdf_values)-1
    low = 0
    mid = round((high+low)/2)
    while not (cdf_values[mid] < percentile <= cdf_values[mid + 1]):
        if cdf_values[mid] > percentile:
            high = mid
        else:
            low = mid
        mid = round((high+low)/2)

    alpha = (percentile - cdf_values[mid]) / (cdf_values[mid+1] - cdf_values[mid])
    return cdf_indices[mid] + alpha * (cdf_indices[mid+1] - cdf_indices[mid])


def evaluate_dice_list(lambda_func, dice_list):
    if isinstance(dice_list,list):
        return [evaluate_dice_list(lambda_func,dice_sublist) for dice_sublist in dice_list]
    else:
        return lambda_func(dice_list)


def get_data_for_plot(dice_list, mode, name_list):
    """

    :param dice_list:
    :param mode:
    :param name_list:
    :return: A list of tuples describing data about each dice.
    First element is x data. Second element is y data. Third element is die string
    """
    data = []
    for i in range(len(dice_list)):
        die = dice_list[i]
        die_string = "{} ({:.2f} / {:.2f})".format(name_list[i], die.mean(), die.std())
        if mode == 'atleast':
            die_dict = sorted(die.at_least().items())
            x = np.array([die_dict[i][0] for i in range(len(die_dict))])
            y = np.array([die_dict[i][1] for i in range(len(die_dict))])
        elif mode == 'atmost':
            die_dict = sorted(die.at_most().items())
            x = np.array([die_dict[i][0] for i in range(len(die_dict))])
            y = np.array([die_dict[i][1] for i in range(len(die_dict))])
        else:
            die_dict = sorted(die.pdf.items())
            x = np.array([die_dict[i][0] for i in range(len(die_dict))])
            y = np.array([die_dict[i][1] for i in range(len(die_dict))])

        data.append((x, y, die_string))

    return data


def update_plot(fig, ax, line_list, data):
    for i in range(len(line_list)):
        line_list[i].set_xdata(data[i][0])
        line_list[i].set_ydata(data[i][1])
        line_list[i].set_label(data[i][2])

    ax.relim()
    ax.autoscale_view()
    ax.legend()
    fig.canvas.draw_idle()


def extract_values_from_sliders(sliders_list):
    return tuple([slider.val for slider in sliders_list])


def print_data(die, dictionary, name=None):
    max_pipes = 150
    filled_char = '█'
    empty_char = ' '
    end_char = '-'
    tabulate.PRESERVE_WHITESPACE = True

    if name is not None:
        die.name = name

    text_header = ['#', '%', "{} ({:.2f} / {:.2f})".format(die.name, die.mean(), die.std())]
    text = [text_header]
    max_pipes = 150
    for roll, chance in sorted(dictionary.items()):
        num_pipes = round(max_pipes * chance)
        progress_bar_string = filled_char * num_pipes + empty_char * (max_pipes - num_pipes) + end_char
        text += [[roll, chance * 100, progress_bar_string]]
    print(tabulate.tabulate(text, headers='firstrow', floatfmt='.2f'))







