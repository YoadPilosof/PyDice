import math

import numpy as np
import copy
import tabulate

import dice


def force_cube(value):
    """
    Force a variable to be a die object
    :param value: The variable to force
    :return: A die object
    """
    # Turn boolean into a number
    if isinstance(value, bool):
        value = value + 0
    # If value is already a die, simply return it
    if isinstance(value, dice.dice):
        return value
    # If value isn't a die, create a die with only one outcome (value)
    else:
        return dice.from_const(value)


def flatten(S):
    """
    Flattens a recursive list (i.e. list of lists of lists ...)
    :param S: The list to flatten
    :return: The flattened list
    """

    # Recursion stop case, an empty list is flattened
    if not S:
        return S

    # If S[0] is a list, we flatten it and the rest of the list and combine the lists
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])

    # If S[0] is not a list, we combine it to the flattened version of the rest of the list
    return S[:1] + flatten(S[1:])


def print_data(die, dictionary, name=None, precision=2):
    """
    Prints the statistics of a die according to a certain mode
    :param die: The die to print
    :param dictionary: The die's data to print. Can be the die's pdf, cdf etc.
    :param name: Optional parameter. If given, sets the die's name and use it for the print
    :param precision: Number describing how many decimal digits the data will have
    """

    # Print parameters
    # Number of pipes corresponding to 100%
    max_pipes = 150
    # How to print a filled bar
    filled_char = '█'
    # How to print an empty bar
    empty_char = ' '
    # Character to place at the end of the bar, for better readability
    end_char = '-'
    # Floating Point Format
    fpf = "." + str(precision) + "f"
    fpf_ext = "{:" + fpf + "}"
    # Force `tabulate` to keep whitespace so empty bars (i.e. very unlikely outcomes) are printed correctly
    tabulate.PRESERVE_WHITESPACE = True

    # If a name was given, update the die's name
    if name is not None:
        die.name = name

    # Table header
    # For the bar column, the header is general information about the die (name, mean, std)
    header = '{} (' + fpf_ext + ' / ' + fpf_ext + ')'
    text_header = ['#', '%', header.format(die.name, die.mean(), die.std())]
    text = [text_header]

    # Loop over all possible rolls, sorted by value
    for roll, chance in sorted(dictionary.items()):
        # The number of pipes to print is proportional to the chance for this outcome
        num_pipes = round(max_pipes * chance)
        # The string of the progress bar
        progress_bar_string = filled_char * num_pipes + empty_char * (max_pipes - num_pipes) + end_char
        # Add a row to the text variable, with:
        # the roll, the chance of the roll (in percentages), and the progress bar string
        text += [[roll, chance * 100, progress_bar_string]]

    # Print the table
    # Use the first row as a header row
    # Force floating point numbers to use 2 decimal places
    print(tabulate.tabulate(text, headers='firstrow', floatfmt=fpf))
    print("")


def find_percentile(cdf_dict, percentile):
    sorted_dict_tuples = sorted(cdf_dict.items())
    cdf_indices = [sorted_dict_tuples[i][0] for i in range(len(sorted_dict_tuples))]
    cdf_values = [sorted_dict_tuples[i][1] for i in range(len(sorted_dict_tuples))]
    if percentile >= cdf_values[-1]:
        return cdf_indices[-1]
    if percentile <= cdf_values[0]:
        return cdf_indices[0]

    high = len(cdf_values) - 1
    low = 0
    mid = round((high + low) / 2)
    while not (cdf_values[mid] < percentile <= cdf_values[mid + 1]):
        if cdf_values[mid] > percentile:
            high = mid
        else:
            low = mid
        mid = round((high + low) / 2)

    alpha = (percentile - cdf_values[mid]) / (cdf_values[mid + 1] - cdf_values[mid])
    return cdf_indices[mid] + alpha * (cdf_indices[mid + 1] - cdf_indices[mid])


def evaluate_dice_list(lambda_func, dice_list):
    """
    Evaluates a generic function on a list of dice
    :param lambda_func: A function that take a die and returns some value
    :param dice_list: A die or a list of dice
    :return: A list describing the return values of each die in dice_list when called with lambda_func
    """

    # If dice_list is a list, call the function on every die in the list
    if isinstance(dice_list, list):
        return [lambda_func(die) for die in dice_list]

    # If dice_list is one die, call the function on it
    else:
        return lambda_func(dice_list)


def extract_values_from_sliders(sliders_list):
    """
    Transforms a list of slider objects to a tuple of their values
    :param sliders_list: A list of slider objects
    :return: A tuple where each element is the value of the corresponding slider
    """
    return tuple([slider.val for slider in sliders_list])


def plot_stats__get_data_for_plot(dice_list, mode):
    """
    Generates a list that describes the data for the dice.plot method
    :param dice_list: A list of dice whose data is required
    :param mode: Which data to return, can be 'normal', 'atleast', or 'atmost'
    :return: A list of tuples describing data about each dice.
    First element is x data. Second element is y data. Third element is die string
    """
    data = []
    for i in range(len(dice_list)):
        die = dice_list[i]
        # Generate a descriptive text to be used in the legend
        # Consists of the die name, and its mean and std
        die_string = "{} ({:.2f} / {:.2f})".format(die.name, die.mean(), die.std())

        # Generate a NumPy array for x and y, depending on the mode
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

        # Append a tuple to the list. The tuple consists of x-data, y-data, and the die_string
        data.append((x, y, die_string))

    return data


def plot_stats__get_x_label(basic_x_label, sliders_list):
    x_label = basic_x_label + '\n('
    for slider in sliders_list:
        x_label += slider.label.get_text() + ' = ' + str(slider.val) + ', '
    return x_label[:-2] + ')'


def plot_mean__get_data_by_slider(dice_list, sliders_list, buttons_list):
    """
    Returns a data structure for the plotting function
    :param dice_list: The list of dice function to evaluate
    :param sliders_list: The list of sliders
    :param buttons_list: The list of buttons
    :return: A data structure containing the x and y values of all the plots, as well as legend and x-labels
    """

    plot_kwargs = {}

    # We first find the index of the pushed button
    pinned = 0
    for i in range(len(buttons_list)):
        if not buttons_list[i].active:
            pinned = i
    # We get the values from all the slider
    sliders_values = list(extract_values_from_sliders(sliders_list))
    # We extract the variable of the pinned slider
    pinned_slider = sliders_list[pinned]
    # We create a list of variables that this slider can get
    # We use these values as the x-axis of the plot
    pinned_values = list(
        range(pinned_slider.valmin, pinned_slider.valmax + pinned_slider.valstep, pinned_slider.valstep))
    die_inst = []
    plot_kwargs['x_data'] = []
    plot_kwargs['y_data'] = []
    plot_kwargs['legend_labels'] = []
    # We loop over all the dice (i.e. all the lines in the plot)
    for die in dice_list:
        # For each one, we create a list:
        # The first element is the values of the x-axis
        # The second element is the values of the y-axis
        # The third element is the name of this die
        dice_by_slider_list = []
        # For each value the pinned slider can get, we evaluate the die and take its average
        for pinned_value in pinned_values:
            # We change the list of variables, only at the index of the pinned slider
            # We change its value according to the current one we're looping over
            sliders_values[pinned] = pinned_value
            # We then create an instantiation of the die by calling it with the arguments
            die_inst = die(*sliders_values)
            # We add the mean to the list of y-axis values
            dice_by_slider_list.append(die_inst.mean())
        plot_kwargs['x_data'].append(pinned_values)
        plot_kwargs['y_data'].append(dice_by_slider_list)
        plot_kwargs['legend_labels'].append(die_inst.name)

    # We also want to return the label of the current pushed button (for the x-axis)
    pinned_label = buttons_list[pinned].label
    x_label_str = pinned_label.get_text()
    if len(sliders_list) > 1:
        x_label_str += '\n('
        for i in range(len(sliders_list)):
            if i != pinned:
                x_label_str += buttons_list[i].label.get_text() + ' = ' + str(sliders_list[i].val) + ', '
        x_label_str = x_label_str[:-2]
        x_label_str += ')'

    plot_kwargs['x_label'] = x_label_str
    plot_kwargs['y_label'] = 'Mean'

    plot_kwargs['x_ticks'] = plot_kwargs['x_data'][0]
    return plot_kwargs


def plot_mean__set_button_callback(buttons_list, button, update_func):
    def button_callback(event):
        for b in buttons_list:
            b.active = True
        button.active = False
        update_func()

    return button_callback


def update_plot(fig, ax, line_list, plot_kwargs):
    """
    Updates a plot with new data
    :param fig: Matplotlib figure object
    :param ax: Matplotlib axes object
    :param line_list: A list of line objects describing the different graphs in the plot
    :param plot_kwargs: A dictionary with the following arguments:
                                x_data: A list of lists that contain the new x-data
                                y_data: A list of lists that contain the new y-data
                                legend_labels: A list of strings describing the label of the legend of each line
                                x_label: The new x-label text
                                y_label: The new y-label text
                                title: New plot title
    """
    # Update x-data, y-data and label for each graph
    for i in range(len(line_list)):
        if 'x_data' in plot_kwargs.keys():
            line_list[i].set_xdata(plot_kwargs['x_data'][i])
        if 'y_data' in plot_kwargs.keys():
            line_list[i].set_ydata(plot_kwargs['y_data'][i])
        if 'legend_labels' in plot_kwargs.keys():
            line_list[i].set_label(plot_kwargs['legend_labels'][i])

    # Remember for each line if it was visible or not. And reset visibility for the legend
    line_vis = []
    for line in line_list:
        line_vis.append(line.get_visible())
        line.set_visible(True)

    # Refresh the legend
    legend_obj = ax.legend()

    # Set visibility to the old value
    for line, vis in zip(line_list, line_vis):
        line.set_visible(vis)

    # Create a dictionary linking the line object in the plot, and the line object in the legend
    lines = {}
    for legend_line, original_line, visible in zip(legend_obj.get_lines(), line_list, line_vis):
        # Set the click tolerance
        legend_line.set_picker(5)
        legend_line.set_alpha(1.0 if visible else 0.2)
        lines[legend_line] = original_line

    # Define the callback method when clicking on a legend item
    def on_pick(event):
        legend_line = event.artist
        original_line = lines[legend_line]
        # Invert the plot visibility
        vis = not original_line.get_visible()
        original_line.set_visible(vis)

        # Set the line's alpha in the legend depending on if it is visible or not
        if vis:
            legend_line.set_alpha(1.0)
        else:
            legend_line.set_alpha(0.2)
        # Redraw the plot after clicking
        fig.canvas.draw()

    # Disconnect all 'pick_event' events from the canvas.
    # These are the callbacks that were attacked in the previous calls to update_plot
    found = fig.canvas.callbacks.callbacks.pop('pick_event', 0)
    # Connect the current event
    cid = fig.canvas.mpl_connect('pick_event', on_pick)

    # Automatically scale the x-axis and y-axis according to the new data
    ax.relim()
    ax.autoscale()
    if 'x_label' in plot_kwargs.keys():
        ax.set_xlabel(plot_kwargs['x_label'])
    if 'y_label' in plot_kwargs.keys():
        ax.set_ylabel(plot_kwargs['y_label'])
    if 'title' in plot_kwargs.keys():
        ax.set_title(plot_kwargs['title'])
    # Set the x-ticks
    if 'x_ticks' in plot_kwargs.keys():
        ax.set_xticks(plot_kwargs['x_ticks'])
    if 'y_ticks' in plot_kwargs.keys():
        ax.set_yticks(plot_kwargs['y_ticks'])
    if 'y_lim' in plot_kwargs.keys():
        y_lim = plot_kwargs['y_lim']
        if y_lim is not None:
            if (isinstance(y_lim, list) or isinstance(y_lim, list)) and len(y_lim) == 2:
                ax.set_ylim(y_lim[0], y_lim[1])
            elif y_lim == 'max':
                current_y_lim = ax.get_ylim()
                if ax.max_y_lim is None:
                    ax.max_y_lim = [current_y_lim[0], current_y_lim[1]]
                ax.max_y_lim[0] = min(ax.max_y_lim[0], current_y_lim[0])
                ax.max_y_lim[1] = max(ax.max_y_lim[1], current_y_lim[1])
                ax.set_ylim(ax.max_y_lim[0], ax.max_y_lim[1])
            else:
                raise Exception('Invalid value for y_lim')
    # Refresh the figure
    fig.canvas.draw_idle()


def generate_all_dice_combs(dice_list):
    """
    Generate a list of all possible combination of values of the list of dice
    :param dice_list: The list of dice to check
    :return: A list of tuples, where the first element is the probability of this outcome
    The second element is the corresponding list of values
    """

    # Initialize all_combs to have one outcome (empty list) with 100% probability
    all_combs = [(1, [])]
    # Loop over each die, recursively updating the list of all combinations
    for each_dice in dice_list:
        # Copy the previous list of all combinations
        old_combs = copy.deepcopy(all_combs)
        all_combs = []
        # Loop over all the combinations we have from the previous dice
        for i in range(len(old_combs)):
            # Loop over all the values of the new die
            for roll, chance in each_dice.pdf.items():
                # The chance to get this new outcome, is the chance to get the list of values of the older dice,
                # times the chance to get the value of the current dice
                new_chance = old_combs[i][0] * chance
                # This outcome is described by a list, of the old values, concatenated with the new value
                new_list = old_combs[i][1] + [roll]
                # Add the new chance-outcome tuple to the list of all combinations
                all_combs += [(new_chance, new_list)]
    return all_combs


def generate_all_ordered_lists_aux(die, N, reverse=False):
    """
    Auxiliary method to be used by generate_all_ordered_lists
    Generates a list of tuples containing all ordered combinations of values of the die
    The first element is the probability of getting this combination (not counting permutations)
    The second element is the list
    The third argument describes how many times this combination can be permuted while being the same
    The fourth argument describes the current run length (i.e. how many times the last item in the list appeared)
    :param die: The die to simulate
    :param N: How many dice were rolled
    :param reverse: Specifies if to sort the list in ascending or descending order
    :return: A list of tuples (float, list)
    """

    # Recursion stop case. For N = 1 (one die), the list of combinations is a list of all possible values
    if N == 1:
        # We wrap each element in a list, because we expect the function to return a list of lists
        return [(chance, [roll], 1, 1) for roll, chance in die.pdf.items()]

    new_combs = []
    # Recursively call the function with length-1
    old_combs = generate_all_ordered_lists_aux(die, N - 1, reverse)
    # Loop over all combinations of the first N-1 dice
    for comb_tuple in old_combs:
        comb_chance = comb_tuple[0]
        comb = comb_tuple[1]
        comb_count = comb_tuple[2]
        comb_run_length = comb_tuple[3]
        # Loop over all values of the Nth die
        for roll, chance in die.pdf.items():
            # If the roll is the same as the last item in the list, it is always valid
            # (regardless of if it is sorted by ascending or descending order)
            # We update the probability of the combination to be
            # the product of the probabilities of the first N-1 elements, and the last element
            # We increment the current run length by 1
            # We update the permutation counter
            if roll == comb[-1]:
                new_combs.append((comb_chance * chance,
                                  comb + [roll],
                                  comb_count * (comb_run_length + 1),
                                  comb_run_length + 1))
            # If the roll is not the same as the last item in the list, we check if it is sorted according to <reverse>
            # We update the probability of the combination to be
            # the product of the probabilities of the first N-1 elements, and the last element
            # We reset the current run length to 1
            # The permutation counter remains the same
            if (roll > comb[-1] and not reverse) or (roll < comb[-1] and reverse):
                new_combs.append((comb_chance * chance, comb + [roll], comb_count, 1))
    return new_combs


def generate_all_ordered_lists(die, N, reverse=False):
    """
    Generates a list of tuples containing all ordered combinations of values of the die
    The first element is the probability of getting this combination (or permutations of this combination)
    The second element is the list
    :param die: The die to simulate
    :param N: How many dice were rolled
    :param reverse: Specifies if to sort the list in ascending or descending order
    :return: A list of tuples (float, list)
    """

    # We first call the auxiliary method
    aux_output = generate_all_ordered_lists_aux(die, N, reverse)
    # We calculate the total number permutations of each combination
    total_combs = math.factorial(N)
    # We want to find how many combinations are permuted to each ordered list
    # So we take to total number of permutations (<total_combs>),
    # and divide it by the number of permutations that leave the list the same
    # (i.e. the third element in the <aux_output> tuple)

    # We process the tuple we got from the auxiliary method
    # by calculating the probability of each combination (with permutation)
    # and by removing unwanted elements in the tuples
    return [(comb_tuple[0] * total_combs / comb_tuple[2], comb_tuple[1]) for comb_tuple in aux_output]
