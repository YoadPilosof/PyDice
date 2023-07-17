from __future__ import annotations

import random
import copy
import math
import dice_utilities
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from matplotlib.widgets import Slider, Button
from typing import Type, Callable, Union


class dice:
    def __init__(self):
        self.pdf = {}
        self.name = "Unnamed Die"

    def __rshift__(self, other: str) -> dice:
        """
        Name a die
        """

        if not isinstance(other, str):
            raise TypeError("Name must be a string")

        self.name = other
        return self

    def __lshift__(self, other: str) -> dice:
        """
        Name a die
        """

        if not isinstance(other, str):
            raise TypeError("Name must be a string")

        self.name = other
        return self

    def print_normal(self, precision: int = 2) -> None:
        """
        Prints general info about the dice (mean and std), and prints a graph-like plot of the pdf of the dice
        """

        if not isinstance(precision, int) or precision < 0:
            raise TypeError("precision must be a non-negative integer")

        dice_utilities.print_data(self, self.pdf, self.name, precision)

    def print_at_least(self, precision: int = 2) -> None:
        """
        Prints general info about the dice (mean and std), and prints a graph-like plot about the chance to get at least a value
        """

        if not isinstance(precision, int) or precision < 0:
            raise TypeError("precision must be a non-negative integer")

        dice_utilities.print_data(self, self.at_least(), self.name, precision)

    def print_at_most(self, precision: int = 2) -> None:
        """
        Prints general info about the dice (mean and std), and prints a graph-like plot about the chance to get at most a value
        """

        if not isinstance(precision, int) or precision < 0:
            raise TypeError("precision must be a non-negative integer")

        dice_utilities.print_data(self, self.at_most(), self.name, precision)

    def at_least(self) -> dict[float, float]:
        """
        :return: A dictionary describing the chance to get at least that result
        """
        new_dict = {}
        old_value = 1
        for roll, chance in sorted(self.pdf.items()):
            new_dict[roll] = old_value
            old_value -= self.pdf[roll]
        return new_dict

    def at_most(self) -> dict[float, float]:
        """
        :return: A dictionary describing the chance to get at most that result
        """
        new_dict = {}
        old_value = 0
        for roll, chance in sorted(self.pdf.items()):
            old_value += self.pdf[roll]
            new_dict[roll] = old_value
        return new_dict

    def mean(self) -> float:
        """
        :return: Returns the mean of the die
        """
        res = 0
        for roll, chance in self.pdf.items():
            res += roll * chance
        return res

    def std(self) -> float:
        """
        :return: Returns the standard deviation of the die
        """

        return math.sqrt(self.var())

    def var(self) -> float:
        """
        :return: Returns the standard deviation of the die
        """

        # Calculate the variance, and later take the square root
        var = 0
        # First find the mean, as it is used in the definition of the variance
        m = self.mean()
        for roll, chance in self.pdf.items():
            var += ((roll - m) ** 2) * chance
        return var

    def max(self) -> float:
        """
        :return: The maximum value the die can roll
        """
        return max(self.pdf.keys())

    def min(self) -> float:
        """
        :return: The minimum value the die can roll
        """
        return min(self.pdf.keys())

    def roll(self, n: int = 0) -> float | list[float]:
        """
        Simulates a roll of the die
        :param n: How many dice rolls to simulate
        :return: The simulates roll(s). If a value was given to n, returns a list, otherwise returns a value
        """

        if not isinstance(n, int) or n < 0:
            raise TypeError("n must be zero (to roll a single die) or a positive integer (to roll a list)")

        # The possible outcomes of the die
        rolls = list(self.pdf.keys())
        # The probability of each outcome
        chances = list(self.pdf.values())

        # Since n=0 describes rolling once and not wrapping it in a list
        # We create a new variable that describes the actual number of rolls to simulate
        n_fix = n if n >= 1 else 1
        # Randomly choose an index (describing a roll), with probabilities according to <chances>
        index = random.choices(range(len(chances)), chances, k=n_fix)
        # If the n argument wasn't given, return a value instead of a list
        if n == 0:
            return rolls[index[0]]
        # For other values of n, convert the list of indices to a list of rolls
        return [rolls[index[i]] for i in range(len(index))]

    def norm(self) -> float:
        """
        Debugging method
        :return: The sum of all probabilities of the die. Should be 1 at all times
        """
        return sum(self.pdf.values())

    def to_fast(self) -> fastdice:
        return fastdice(self.mean(), self.var()) << self.name

    # ~~~~~~~~~ Overloaded Binary and Unary Operations ~~~~~~~~~

    def __add__(self, other: dice | float | int) -> dice:
        """
        Creates a die describing the sum of two dice
        :param other: The second die to add. Can be a number
        :return: A new die, with statistics according to the sum of the two dice
        """

        other = dice_utilities.force_cube(other)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The outcome of these two rolls, is the sum of the rolls
                new_roll = self_roll + other_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[new_roll] += self_chance * other_chance

        # Return the new die
        return new_dice

    def __sub__(self, other: dice | float | int) -> dice:
        """
        Creates a die describing the difference of two dice
        :param other: The second die to subtract. Can be a number
        :return: A new die, with statistics according to the difference of the two dice
        """

        other = dice_utilities.force_cube(other)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The outcome of these two rolls, is the difference of the rolls
                new_roll = self_roll - other_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[new_roll] += self_chance * other_chance

        # Return the new die
        return new_dice

    def __mul__(self, other: dice | float | int) -> dice:
        """
        Creates a die describing the product of two dice
        :param other: The second die to multiply. Can be a number
        :return: A new die, with statistics according to the product of the two dice
        """

        other = dice_utilities.force_cube(other)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The outcome of these two rolls, is the product of the rolls
                new_roll = self_roll * other_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[new_roll] += self_chance * other_chance

        # Return the new die
        return new_dice

    def __truediv__(self, other: dice | float | int) -> dice:
        """
        Creates a die describing the floating point division of two dice
        :param other: The second die to divide by. Can be a number
        :return: A new die, with statistics according to the division of the two dice
        """

        other = dice_utilities.force_cube(other)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The outcome of these two rolls, is the division of the rolls
                new_roll = self_roll / other_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[new_roll] += self_chance * other_chance

        # Return the new die
        return new_dice

    def __mod__(self, other: dice | float | int) -> dice:
        """
        Creates a die describing the modulus of two dice
        :param other: The second die to perform modulus by. Can be a number
        :return: A new die, with statistics according to the modulus of the two dice
        """

        other = dice_utilities.force_cube(other)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The outcome of these two rolls, is the modulus of the rolls
                new_roll = self_roll % other_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[new_roll] += self_chance * other_chance

        # Return the new die
        return new_dice

    def __floordiv__(self, other: dice | float | int) -> dice:
        """
        Creates a die describing the integer division of two dice
        :param other: The second die to divide by. Can be a number
        :return: A new die, with statistics according to the division of the two dice
        """

        other = dice_utilities.force_cube(other)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The outcome of these two rolls, is the integer division of the rolls
                new_roll = self_roll // other_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[new_roll] += self_chance * other_chance

        # Return the new die
        return new_dice

    def __pow__(self, power: int | dice, modulo=None) -> dice:
        """
        Creates a die describing rolling the same die multiple times and taking the sum
        :param power: How many dice to roll
        :return: A new die, with statistics according to the sum of the die
        """

        if isinstance(power, dice):
            new_dice = dice()

            for power_roll, power_chance in power.pdf.items():
                temp_die = self ** power_roll
                for temp_roll, temp_chance in temp_die.pdf.items():
                    if temp_roll not in new_dice.pdf.keys():
                        new_dice.pdf[temp_roll] = 0
                    new_dice.pdf[temp_roll] += power_chance * temp_chance

            return new_dice

        if not isinstance(power, int) or power < 0:
            raise TypeError("the power operator (rolling multiple dice) is only supported for non-negative integers and dice that roll non-negative integers")

        # Roll a die zero times, so the result is always 0
        if power == 0:
            return zero()
        # Roll a die once, so copy the self die and return it
        elif power == 1:
            return self
        # Roll more than one die, call power recursively
        else:
            # return self ** (power/2) ** 2
            recursive_die = self ** (power // 2)
            if power % 2:
                return recursive_die + recursive_die + self
            else:
                return recursive_die + recursive_die

    def __neg__(self) -> dice:
        """
        :return: Flips the sign of the values of the dice
        """
        new_dice = dice()
        for roll, chance in self.pdf.items():
            new_dice.pdf[-roll] = chance
        return new_dice

    # ~~~~~~~~~ Overloaded Right side Operations ~~~~~~~~~

    def __radd__(self, other: float | int) -> dice:

        if not isinstance(other, (float, int)):
            raise TypeError("Addition can only be done between a die and another die / number")

        return self + other

    def __rsub__(self, other: float | int) -> dice:

        if not isinstance(other, (float, int)):
            raise TypeError("Subtraction can only be done between a die and another die / number")

        return (-self) + other

    def __rmul__(self, other: float | int) -> dice:

        if not isinstance(other, (float, int)):
            raise TypeError("Multiplication can only be done between a die and another die / number")

        return self * other

    def __rtruediv__(self, other: float | int) -> dice:

        if not isinstance(other, (float, int)):
            raise TypeError("Division can only be done between a die and another die / number")

        return from_const(other) / self

    def __rfloordiv__(self, other: float | int) -> dice:

        if not isinstance(other, (float, int)):
            raise TypeError("Division can only be done between a die and another die / number")

        return from_const(other) // self

    # ~~~~~~~~~ Common die manipulation ~~~~~~~~~

    def adv(self, n: int = 2) -> dice:
        """
        :return: A new die, describing rolling the current die n times, and taking the highest result
        """

        if not isinstance(n, int) or n < 1:
            raise TypeError("n must be a positive integer")

        if n == 1:
            return self
        else:
            recursive_die = self.adv(n // 2)
            if n % 2 == 0:
                return highest(recursive_die, recursive_die)
            else:
                return highest(recursive_die, recursive_die, self)

    def dis(self, n: int = 2) -> dice:
        """
        :return: A new die, describing rolling the current die n times, and taking the lowest result
        """

        if not isinstance(n, int):
            raise TypeError("n must be a positive integer")

        if n == 1:
            return self
        else:
            recursive_die = self.dis(n // 2)
            if n % 2 == 0:
                return lowest(recursive_die, recursive_die)
            else:
                return lowest(recursive_die, recursive_die, self)

    def exp(self, explode_on: float | int | None = None, max_depth: int = 2) -> dice:
        """
        Simulates exploding a die, i.e., rolling the die, and if it rolls the highest value, roll it again
        :param explode_on: Optional parameter. If given, the die will explode on this value, instead of the maximum value of the die
        :param max_depth: Maximum depth of the simulation. Default is 2
        :return: A new die
        """

        if not isinstance(explode_on, (float, int) or explode_on is None):
            raise TypeError("explode_on must be a number or None")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise TypeError("max_depth must be a non-negative integer")

        if explode_on is None:
            explode_on = self.max()

        # Recursion stop case, if max_depth is zero, we never roll another die, so we simply return the <self> die
        if max_depth == 0:
            return self

        # Recursively call exp with a smaller max_depth
        deeper_dice = self.exp(max_depth=max_depth - 1)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            # If the value roll for the <self> die is the value it explodes on, consider the <deeper_dice> die
            if self_roll == explode_on:
                for deeper_dice_roll, deeper_dice_chance in deeper_dice.pdf.items():
                    # The outcome of these two rolls, is the sum of the rolls
                    new_roll = self_roll + deeper_dice_roll
                    # If this roll is not a possible roll of the new dice, add it with probability of 0
                    if not (new_roll in new_dice.pdf):
                        new_dice.pdf[new_roll] = 0
                    # Increase the chance of getting this outcome by the product of the probabilities of each die
                    new_dice.pdf[new_roll] += self_chance * deeper_dice_chance

            # If the <self> die did not explode, ignore the <deeper_dice> die
            else:
                # The outcome of these two rolls, is only first value rolled (since it did not explode)
                new_roll = self_roll
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                # Increase the chance of getting this outcome by the product of the probability of the first die
                new_dice.pdf[new_roll] += self_chance

        return new_dice

    def count(self, *args) -> dice:
        """
        Simulates rolling a die, and checking if it appears in the list.
        The value is based on the amount of times the roll appears in the list
        :param args: A number or a list to compare. Can be a nested list
        :return: A new die
        """

        # Transform the tuple to a list
        match_list = list(args)

        # Flatten the nested list
        flattened_match_list = dice_utilities.flatten(match_list)

        new_dice = dice()

        for roll, chance in self.pdf.items():
            # The roll of the new die is how many times the roll of the <self> die appears the match list
            new_roll = flattened_match_list.count(roll)
            # If this roll is not a possible roll of the new dice, add it with probability of 0
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += chance

        return new_dice

    def count_attempts(self, max_depth: int) -> dice:
        """
        Simulates rolling the self die until getting a non-zero result.
        :param max_depth: Maximum simulation depth, after max_depth rolls, we assume the next one is non-zero
        :return: The chance to need N rolls until a non-zero value is rolled
        """

        if not isinstance(max_depth, int) or max_depth < 1:
            raise TypeError("max_depth must be a positive integer")

        q = self.pdf[0]
        p = 1 - q
        new_die = dice()
        for i in range(max_depth):
            new_die.pdf[i + 1] = math.pow(q, i) * p
        new_die.pdf[max_depth + 1] = math.pow(q, max_depth)

        return new_die

    def cond(self, die_if_true: dice | float | int, die_if_false: dice | float | int = 0) -> dice:
        """
        Uses the first boolean die to choose a second die
        :param die_if_true: The die to choose if the first one is True
        :param die_if_false: The die to choose if the first one is False
        :return: The new die
        """

        die_if_true = dice_utilities.force_cube(die_if_true)
        die_if_false = dice_utilities.force_cube(die_if_false)

        new_dice = dice()

        # First loop over all the rolls of the <die_if_false> die
        # Since it is the first outcome we test, we don't need to see if it is already in the dictionary of the new die
        for roll, chance in die_if_false.pdf.items():
            # The chance to get this roll is the chance to get a false value (self.pdf[0])
            # times the chance to get the roll given a false value (chance)
            new_dice.pdf[roll] = chance * self.pdf[0]

        # Loop over all the rolls of the <die_if_true> die
        for roll, chance in die_if_true.pdf.items():
            # We now need to check if this roll is in the dictionary. If not, create it with zero probability
            if roll not in new_dice.pdf.keys():
                new_dice.pdf[roll] = 0
            # The chance to get this roll is the chance to get a true value (self.pdf[1])
            # times the chance to get the roll given a true value (chance)
            new_dice.pdf[roll] += chance * self.pdf[1]

        return new_dice

    def switch(self, *args: dice | float | int, default=None) -> dice:
        """
        Uses the first die (which rolls integers between 0 and N-1) to choose a second die
        :param args: A list or tuple of the other set of dice
        :param default: A value or dice to take when the self die gets a value not described by args.
                        If this is None, the function will throw an error.
        :return: The new die
        """

        # Create a variable called <other_dice_list> which is a list
        if len(args) == 1:
            # If the user gave one argument, which is a list, use it as the list
            if isinstance(args[0], list):
                other_dice_list = args[0]
            # If the user gave one argument, which is a value, treat it as a list of length 1
            else:
                other_dice_list = [args[0]]
        else:
            # Otherwise, turn the tuple into a list
            other_dice_list = list(args)

        # Force each element in the list to be a die
        for i in range(len(other_dice_list)):
            other_dice_list[i] = dice_utilities.force_cube(other_dice_list[i])

        if default is not None:
            default = dice_utilities.force_cube(default)

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            if self_roll < len(other_dice_list):
                for other_roll, other_chance in other_dice_list[self_roll].pdf.items():
                    # If this roll is not a possible roll of the new dice, add it with probability of 0
                    if other_roll not in new_dice.pdf.keys():
                        new_dice.pdf[other_roll] = 0
                    # Increase the chance of getting this outcome by the product of the probabilities of each die
                    new_dice.pdf[other_roll] += self_chance * other_chance
            else:
                if default is None:
                    raise Exception('The self die can get a value of ' + str(
                        self_roll) + ' which is not described by the args list')
                else:
                    for default_roll, default_chance in default.pdf.items():
                        # If this roll is not a possible roll of the new dice, add it with probability of 0
                        if default_roll not in new_dice.pdf.keys():
                            new_dice.pdf[default_roll] = 0
                        # Increase the chance of getting this outcome by the product of the probabilities of each die
                        new_dice.pdf[default_roll] += self_chance * default_chance

        return new_dice

    def dictionary(self, roll_dictionary: dict[float, dice | float | int], default_value=None) -> dice:
        """
        Transforms the die according to a dictionary
        :param roll_dictionary: A dictionary describing the transformation, has <float> input and <die>/<float> output
        :param default_value: Optional parameter.
        If given, values that do not have a dictionary value will get this value instead
        :return: The new die
        """

        if default_value is not None:
            default_value = dice_utilities.force_cube(default_value)

        new_dice = dice()
        # Loop over all the values the self die can roll
        for roll, chance in self.pdf.items():
            # Check if the current roll is an entry in the dictionary
            # If it is, use the dictionary. If not, use the default value or raise an error
            if roll in roll_dictionary.keys():
                new_value = roll_dictionary[roll]
                # Transform the new roll into a die if it is not already
                new_value = dice_utilities.force_cube(new_value)
                for new_roll, new_chance in new_value.pdf.items():
                    if not (new_roll in new_dice.pdf):
                        new_dice.pdf[new_roll] = 0
                    new_dice.pdf[new_roll] += chance * new_chance
            else:
                if default_value is None:
                    # If the dictionary doesn't have information about this roll, and there is no default value,
                    # raise an error
                    raise Exception('Value of ' + str(roll) + ' is not described by the dictionary')
                else:
                    for new_roll, new_chance in default_value.pdf.items():
                        if not (new_roll in new_dice.pdf):
                            new_dice.pdf[new_roll] = 0

        return new_dice

    def ranges(self, *args, upper: bool = True) -> dice:
        """
        Gives a value according to the range, such that:
        ( -∞, args[0] ] → 0, ( args[0], args[1] ] → 1, ⋯ ( args[N-1], +∞ ) → N
        :param args: Comparison bins
        :param upper: If set to true, values that are exactly on an edge, will be mapped to the upper bin,
                      i.e., the bins are left-closed right-open [a, b).
                      If set to false,values that are exactly on an edge, will be mapped to the lower bin,
                      i.e., the bins are right-closed left-open (a, b].
        :return: A new die, with values 0 ⋯ N, describing the chance to fall within each bin
        """

        if len(args) == 0:
            raise TypeError("Missing input arguments")

        if len(args) == 1:
            if isinstance(args[0], list):
                edges = args[0]
            elif isinstance(args[0], (float, int)):
                edges = [args[0]]
            else:
                raise TypeError("Input arguments must numbers or a single list")
        else:
            edges = list(args)

        if sorted(edges) != edges:
            raise TypeError("Given bin edges must be in ascending order")

        edges += [float('inf')]

        new_dice = dice()

        current_edge_idx = 0
        for self_roll, self_chance in sorted(self.pdf.items()):
            if upper:
                while self_roll >= edges[current_edge_idx]:
                    current_edge_idx += 1
            else:
                while self_roll > edges[current_edge_idx]:
                    current_edge_idx += 1

            if not (current_edge_idx in new_dice.pdf):
                new_dice.pdf[current_edge_idx] = 0
            new_dice.pdf[current_edge_idx] += self_chance

        return new_dice

    def get_pos(self, pos_index: int | list[int], num_dice: int) -> dice:
        """
        Simulates rolling the <self> die <num_dice> times, and taking the <pos_index>th largest value
        :param pos_index: Which die to take. Can be a list
        :param num_dice: How many dice were rolled
        :return: The new die
        """

        # Transform the pos_index so negative values are counted from the end of the list
        if isinstance(pos_index, list):
            pos_index = [index % num_dice for index in pos_index]
        elif isinstance(pos_index, int):
            pos_index = pos_index % num_dice
        else:
            raise TypeError("pos_index must be an integer or a list of integers")

        # If pos_index is a list, calculate the PDF by looping over all ordered combinations
        if isinstance(pos_index, list):
            new_dice = dice()
            combs = dice_utilities.generate_all_ordered_lists(self, num_dice, True)
            for comb_tuple in combs:
                chance = comb_tuple[0]
                new_roll = sum([comb_tuple[1][index] for index in pos_index])

                if new_roll not in new_dice.pdf.keys():
                    new_dice.pdf[new_roll] = 0
                new_dice.pdf[new_roll] += chance

        # If pos_index is a value, calculate the PDF mathematically
        else:
            pdf = self.pdf
            cdf = self.at_most()
            new_dice = dice()
            n = num_dice
            # In maths, the first order statistics is the lowest value, not highest
            # So we change the pos_index variable accordingly
            r = n - pos_index
            for roll, chance in pdf.items():
                sum_res = 0
                p1 = cdf[roll] - pdf[roll]  # P(X < roll)
                p2 = pdf[roll]  # P(X = roll)
                p3 = 1 - cdf[roll]  # P(X > roll)
                for j in range(0, n - r + 1):
                    norm_fact = math.comb(n, j)
                    factor1 = p3 ** j * ((p1 + p2) ** (n - j))
                    factor2 = (p2 + p3) ** j * (p1 ** (n - j))
                    sum_res += norm_fact * (factor1 - factor2)
                new_dice.pdf[roll] = sum_res

        return new_dice

    def drop_pos(self, pos_index: int | list[int], num_dice: int) -> dice:
        """
        Simulates rolling the <self> die <num_dice> times, and dropping the <pos_index>th largest value
        :param pos_index: Which die to drop. Can be a list
        :param num_dice: How many dice were rolled
        :return: The new die
        """

        # Turn pos_index to a list (if it is not already)
        if not isinstance(pos_index, list):
            pos_index = [pos_index]
        # We first perform modulus on each index, so -1 will turn to num_dice-1
        mod_pos_index = [index % num_dice for index in pos_index]
        # We convert the list of indices to drop, to a list of indices to keep
        keep_pos_index = [i for i in range(num_dice) if not (i in mod_pos_index)]
        # We call the get_pos method with the list of indices to keep
        return self.get_pos(keep_pos_index, num_dice)

    def round(self) -> dice:
        """
        Rounds the die result
        :return: The new die
        """

        new_dice = dice()

        for roll, chance in self.pdf.items():
            new_roll = round(roll)
            # If this roll is not a possible roll of the new dice, add it with probability of 0
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            # Increase the chance of getting this outcome by the product of the probabilities of each die
            new_dice.pdf[new_roll] += chance

        # Return the new die
        return new_dice

    def floor(self) -> dice:
        """
        Rounds down the die result
        :return: The new die
        """

        new_dice = dice()

        for roll, chance in self.pdf.items():
            new_roll = math.floor(roll)
            # If this roll is not a possible roll of the new dice, add it with probability of 0
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            # Increase the chance of getting this outcome by the product of the probabilities of each die
            new_dice.pdf[new_roll] += chance

        # Return the new die
        return new_dice

    def ceil(self) -> dice:
        """
        Rounds up the die result
        :return: The new die
        """

        new_dice = dice()

        for roll, chance in self.pdf.items():
            new_roll = math.ceil(roll)
            # If this roll is not a possible roll of the new dice, add it with probability of 0
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            # Increase the chance of getting this outcome by the product of the probabilities of each die
            new_dice.pdf[new_roll] += chance

        # Return the new die
        return new_dice

    def func(self, lambda_func: Callable[[Union[float, int]], Union[float, int, dice]]) -> dice:
        """
        Executes a generic, 1 argument function, on the <self> die
        :param lambda_func: The function to execute
        :return: The new die
        """

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            value = lambda_func(self_roll)
            value = dice_utilities.force_cube(value)
            for value_roll, value_chance in value.pdf.items():
                if value_roll not in new_dice.pdf.keys():
                    new_dice.pdf[value_roll] = 0
                new_dice.pdf[value_roll] += self_chance * value_chance

        return new_dice

    # ~~~~~~~~~ Overloaded Comparative Operations ~~~~~~~~~

    def __lt__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is less than <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll < other_roll) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __le__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is less than or equal to <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll <= other_roll) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __eq__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is equal to <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll == other_roll) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __ne__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is not equal to <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll != other_roll) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __gt__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is greater than <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll > other_roll) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __ge__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is greater than or equal to <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll >= other_roll) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    # ~~~~~~~~~ Overloaded Boolean Operators ~~~~~~~~~

    def __and__(self, other: dice) -> dice:
        """
        Creates a boolean die, describing the 'and' operation on two dice
        Each die is treated as a boolean die, where a value of 0 is treated as false, and any other value is treated as true
        :param other: The second die
        :return: The new die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # We turn the rolls to a boolean value
                # A roll of 0 is treated as false, and any other roll is treated as true
                self_roll_bool = self_roll != 0
                other_roll_bool = other_roll != 0
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll_bool and other_roll_bool) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __or__(self, other: dice) -> dice:
        """
        Creates a boolean die, describing the 'or' operation on two dice
        Each die is treated as a boolean die, where a value of 0 is treated as false, and any other value is treated as true
        :param other: The second die
        :return: The new die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # We turn the rolls to a boolean value
                # A roll of 0 is treated as false, and any other roll is treated as true
                self_roll_bool = self_roll != 0
                other_roll_bool = other_roll != 0
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll_bool or other_roll_bool) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __xor__(self, other: dice) -> dice:
        """
        Creates a boolean die, describing the 'xor' operation on two dice
        Each die is treated as a boolean die, where a value of 0 is treated as false, and any other value is treated as true
        :param other: The second die
        :return: The new die
        """

        other = dice_utilities.force_cube(other)

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other.pdf.items():
                # We turn the rolls to a boolean value
                # A roll of 0 is treated as false, and any other roll is treated as true
                self_roll_bool = self_roll != 0
                other_roll_bool = other_roll != 0
                # The new roll is the comparison result of the <self> and <other> die.
                # We add 0 to force the comparison to return a number 0/1 and not False/True
                new_roll = (self_roll_bool != other_roll_bool) + 0
                new_dice.pdf[new_roll] += self_chance * other_chance

        return new_dice

    def __invert__(self) -> dice:
        """
        Creates a boolean die, describing the inversion operator on a die
        The die is treated as a boolean die, where a value of 0 is treated as false, and any other value is treated as true
        :return: The new die
        """

        # Since we return a boolean die, we know it can only roll 0 [false] or 1 [true]
        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0
        for roll, chance in self.pdf.items():
            # We treat a roll of 0 on the <self> die as false, so after inversion, we return 1
            # We treat any other roll on the <self> die as true, so after inversion, we return 0
            new_roll = 1 if roll == 0 else 0
            new_dice.pdf[new_roll] = chance
        return new_dice

    # ~~~~~~~~~ Reroll Methods ~~~~~~~~~

    def reroll_func(self, lambda_cond: Callable[[float | int], bool], max_depth: int = 1) -> dice:
        """
        A general reroll function. Simulates rolling a die, and rerolling the dice if a certain condition is met
        :param lambda_cond: A lambda function that takes a number and returns a boolean.
        :param max_depth: A parameter describing if the dice is rerolled multiple times if a bad result is continuously rolled
        Can be an integer (1 for only one reroll), or 'inf' for an infinite number of rerolls
        :return: A new die
        """

        prob_rerollable = sum([chance for roll, chance in self.pdf.items() if lambda_cond(roll)])
        prob_staying = sum([chance for roll, chance in self.pdf.items() if not lambda_cond(roll)])

        if max_depth == 'inf':
            final_prob_rerollable = 0
            final_prob_staying = 1
        else:
            final_prob_rerollable = pow(prob_rerollable, max_depth)
            final_prob_staying = 1 - final_prob_rerollable

        new_dice = dice()

        for roll, chance in self.pdf.items():
            if lambda_cond(roll):
                new_dice.pdf[roll] = chance * final_prob_rerollable
            else:
                new_dice.pdf[roll] = (chance / prob_staying) * final_prob_staying + (chance / 1) * final_prob_rerollable

        return new_dice

    def reroll_on(self, match_list: float | int | list[float | int], max_depth: int = 1) -> dice:
        """
        Simulates rolling a die, and rerolling the dice if the value is in match_list
        :param match_list: A number or a list to compare. Can be a nested list
        :param max_depth: A parameter describing if the dice is rerolled multiple times if a bad result is continuously rolled
        Can be an integer (1 for only one reroll), or 'inf' for an infinite number of rerolls
        :return: A new die
        """

        # If match_list is not a list, but a value, force it to be a list of length 1
        if not isinstance(match_list, list):
            match_list = [match_list]

        # Flatten the match list, so it is not nested
        flattened_match_list = dice_utilities.flatten(match_list)

        # Call the general reroll method, with the reroll condition being if the roll was on the reroll list
        return self.reroll_func(lambda x: x in flattened_match_list, max_depth)

    def reroll_comp(self, sign: str, val: dice | float | int, max_depth: int = 1) -> dice:
        """
        Rerolls the <self> die based on a comparison condition
        :param sign: A string defining the comparison
        :param val: The value to compare to
        :param max_depth: A parameter describing if the dice is rerolled multiple times if a bad result is continuously rolled
        Can be an integer (1 for only one reroll), or 'inf' for an infinite number of rerolls
        :return: The new die
        """

        match sign:
            case '>':
                return self.reroll_func(lambda x: x > val, max_depth=max_depth)
            case '>=':
                return self.reroll_func(lambda x: x >= val, max_depth=max_depth)
            case '<':
                return self.reroll_func(lambda x: x < val, max_depth=max_depth)
            case '<=':
                return self.reroll_func(lambda x: x <= val, max_depth=max_depth)
            case '==':
                return self.reroll_func(lambda x: x == val, max_depth=max_depth)
            case '!=':
                return self.reroll_func(lambda x: x != val, max_depth=max_depth)
            # If the sign is not one of those six, raise an error
            case _:
                raise Exception('Invalid comparison operator')

    def given(self, lambda_cond: Callable[[float | int], bool]) -> dice:
        """
        Changes the values of the self die to be given a conditional probability
        :param lambda_cond: A function that takes roll values and gives a boolean.
        :return: A new die that has the properties of the original die, given that lambda_cond is true
        """

        P_B = sum([chance for roll, chance in self.pdf.items() if lambda_cond(roll)])
        if P_B == 0:
            raise AttributeError("The self die never gets a value where the lambda condition is true")

        new_dice = dice()

        for roll, chance in self.pdf.items():
            if lambda_cond(roll):
                new_dice.pdf[roll] = chance / P_B

        return new_dice


class fastdice:
    def __init__(self, mean: float | int = 0, var: float | int = 0):
        self._mean = mean
        self._var = var
        self.name = "Unnamed Die"

    def __rshift__(self, other: str) -> fastdice:
        """
        Name a die
        """
        self.name = other
        return self

    def __lshift__(self, other: str) -> fastdice:
        """
        Name a die
        """
        self.name = other
        return self

    def print(self, precision: int = 2) -> None:
        """
        Prints information about the dice
        """

        txt_format = "{0}: {1:." + str(precision) + "f} ± {2:." + str(precision) + "f}"
        print(txt_format.format(self.name, self._mean, math.sqrt(self._var)))

    def mean(self) -> float:
        """
        :return: Returns the mean of the die
        """

        return self._mean

    def std(self) -> float:
        """
        :return: Returns the standard deviation of the die
        """

        return math.sqrt(self._var)

    def var(self) -> float:
        """
        :return: Returns the variance of the die
        """

        return self._var

    def to_dice(self, step: float | int = 1, span: float | int = 5) -> dice:

        if not isinstance(step, (float, int)):
            raise TypeError("step must be a number")
        if not isinstance(span, (float, int)):
            raise TypeError("span must be a number")

        new_dice = dice()
        m = self.mean()
        v = self.var()
        s = self.std()

        left_idx = math.floor((m - s * span) / step)
        right_idx = math.ceil((m + s * span) / step) + 1  # Since range is exclusive for stop value

        for val in range(left_idx, right_idx):
            new_dice.pdf[val * step] = step / (s * math.sqrt(2 * math.pi)) * math.exp(-(val * step - m) ** 2 / (2 * v))

        n = 1 / new_dice.norm()

        for roll in new_dice.pdf.keys():
            new_dice.pdf[roll] *= n

        return new_dice << self.name

    # ~~~~~~~~~ Overloaded Binary and Unary Operations ~~~~~~~~~

    def __add__(self, other: fastdice | dice | float | int) -> fastdice:
        """
        Creates a die describing the sum of two dice
        :param other: The second die to add. Can be a number
        :return: A new die, with statistics according to the sum of the two dice
        """

        o_mean, o_var = dice_utilities.get_mean_and_var(other)
        new_dice = fastdice(self.mean() + o_mean, self.var() + o_var)
        return new_dice

    def __sub__(self, other: fastdice | dice | float | int) -> fastdice:
        """
        Creates a die describing the difference of two dice
        :param other: The second die to subtract. Can be a number
        :return: A new die, with statistics according to the difference of the two dice
        """

        o_mean, o_var = dice_utilities.get_mean_and_var(other)
        new_dice = fastdice(self.mean() - o_mean, self.var() + o_var)
        return new_dice

    def __mul__(self, other: fastdice | dice | float | int) -> fastdice:
        """
        Creates a die describing the product of two dice
        :param other: The second die to multiply. Can be a number
        :return: A new die, with statistics according to the product of the two dice
        """

        o_mean, o_var = dice_utilities.get_mean_and_var(other)
        new_var = (self.mean() ** 2 + self.var()) * (o_mean ** 2 + o_var) - (self.mean() * o_mean) ** 2
        new_dice = fastdice(self.mean() * o_mean, new_var)
        return new_dice

    def __pow__(self, power: float | int, modulo=None) -> fastdice:
        """
        Creates a die describing rolling the same die multiple times and taking the sum
        :param power: How many dice to roll
        :return: A new die, with statistics according to the sum of the die
        """

        if not isinstance(power, (float, int)):
            raise TypeError("the power operator (rolling multiple dice) is only supported for non-negative integers")

        self._mean *= power
        self._var *= power
        return self

    def __neg__(self) -> fastdice:
        """
        :return: Flips the sign of the values of the dice
        """
        self._mean = -self._mean
        return self

    # ~~~~~~~~~ Overloaded Right side Operations ~~~~~~~~~

    def __radd__(self, other: dice | float | int) -> fastdice:
        return self + other

    def __rsub__(self, other: dice | float | int) -> fastdice:
        return (-self) + other

    def __rmul__(self, other: dice | float | int) -> fastdice:
        return self * other

    # ~~~~~~~~~ Overloaded Comparative Operations ~~~~~~~~~

    def __lt__(self, other: fastdice | dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is less than <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        return (self > other).__invert__()

    def __le__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is less than or equal to <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        return (self > other).__invert__()

    def __gt__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is greater than <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        if isinstance(other, int) or isinstance(other, float):
            chance = math.erfc((other - self.mean()) / (math.sqrt(2) * self.std())) / 2
            return binary(chance)

        if isinstance(other, fastdice):
            return self - other < 0

        if isinstance(other, dice):
            total_chance = 0
            for other_roll, other_chance in other.pdf.items():
                current_chance = math.erfc((other_roll - self.mean()) / (math.sqrt(2) * self.std())) / 2
                total_chance += current_chance * other_chance
            return binary(total_chance)

        raise TypeError("Comparison operators are only compatible between numbers, dice objects and fastdice objects")

    def __ge__(self, other: dice | float | int) -> dice:
        """
        Creates a boolean die (can roll 0 [false] or 1 [true])
        The new die describes if <self> is greater than or equal to <other>
        :param other: The second die to compare <self> to. Can be a number
        :return: The new boolean die
        """

        return self > other


# ~~~~~~~~~ Custom Dice Constructors ~~~~~~~~~


def d(size: int, n: int = 1) -> dice:
    """
    Constructs a new die, which is uniformly distributed over 1 .. size
    :param size: The size of the die
    :param n: How many dice are rolled
    :return: The new die
    """

    if not isinstance(n, int) or n < 1:
        raise TypeError("n must be a positive integer")
    if not isinstance(size, int) or size < 1:
        raise TypeError("size must be a positive integer")

    new_dice = dice()
    if n == 1:
        for i in range(size):
            new_dice.pdf[i + 1] = 1.0 / size

    # This is not optimized, you should use d(k)**n
    else:
        for s in range(n, size * n + 1):
            curr_sum = 0
            factor0 = 1  # Running calculation of (-1)**k
            factor1 = 1  # Running calculation of comb(n,k)
            for k in range((s - n) // size + 1):
                curr_sum += factor0 * factor1 * math.comb(s - size * k - 1, n - 1)
                factor0 *= -1
                factor1 *= n - k
                factor1 //= k + 1
            new_dice.pdf[s] = curr_sum / (size ** n)
    return new_dice


def standard_dice() -> tuple[dice, dice, dice, dice, dice, dice, dice]:
    """
    A helpful function that generates the 7 basic RPG dice. Use:
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    :return: The basic 7 dice
    """
    d4 = d(4) << "d4"
    d6 = d(6) << "d6"
    d8 = d(8) << "d8"
    d10 = d(10) << "d10"
    d12 = d(12) << "d12"
    d20 = d(20) << "d20"
    d100 = d(100) << "d100"

    return d4, d6, d8, d10, d12, d20, d100


def from_const(val: int | float) -> dice:
    """
    Constructs a trivial die with only one value
    :param val: The value of the die
    :return: The new die
    """

    if not isinstance(val, (float, int)):
        raise TypeError("Val must be a number")

    new_dice = dice()
    new_dice.pdf[val] = 1
    return new_dice


def zero() -> dice:
    """
    Constructs a trivial die with only value of 0
    :return: The new die
    """
    new_dice = dice()
    new_dice.pdf[0] = 1
    return new_dice


def list2dice(values_list: list[float | int]) -> dice:
    """
    Constructs a new die according to a list, such that the chance to get each value in the list is equiprobable
    If a value appears multiple times in the list, the chance of rolling it increases accordingly
    :param values_list: The list describing the new dice
    :return: The new die
    """
    new_dice = dice()
    for elem in values_list:
        # If elem isn't in the new die dictionary, set its probability to 0
        if not (elem in new_dice.pdf):
            new_dice.pdf[elem] = 0
        # Since all rolls are equiprobable, the chance to get any result is 1 / total number of options
        new_dice.pdf[elem] += 1 / len(values_list)

    return new_dice

    # ~~~~~~~~~ Dice Functions ~~~~~~~~~


def range2dice(*args: float) -> dice:
    """
    Creates a die with the possible results of 0 .. N-1 with probability according to the input
    :param args: The weights of the different values
    :return: The new die
    """

    # Create a variable called <chance_list> which is a list
    if len(args) == 1:
        # If the user gave one argument, which is a list, use it as the list
        if isinstance(args[0], list):
            chance_list = args[0]
        # If the user gave one argument, which is a value, treat it as a list of length 1
        else:
            chance_list = [args[0]]
    else:
        # Otherwise, turn the tuple into a list
        chance_list = list(args)

    # We normalize all probabilities so the sum of probabilities is 1
    norm_factor = sum(chance_list)
    chance_list_norm = [chance / norm_factor for chance in chance_list]

    new_dice = dice()
    for i in range(len(chance_list_norm)):
        new_dice.pdf[i] = chance_list_norm[i]

    return new_dice


def binary(chance: float) -> dice:
    """
    Creates a die with results of 0 or 1
    :param chance: The chance to get 1
    :return: The new die
    """

    if not isinstance(chance, float) or not (0 < chance < 1):
        raise TypeError("chance must be a number between 0 and 1")

    new_dice = dice()
    new_dice.pdf[0] = 1 - chance
    new_dice.pdf[1] = chance
    return new_dice


# ~~~~~~~~~ Other Dice Methods ~~~~~~~~~


def get_pos(dice_list: list[dice | float | int], index_list: int | list[int] | None = None) -> dice | list[dice]:
    """
    Simulates the result of rolling multiple dice, sorting them, taking values after the sorting, and summing the dice
    :param dice_list: A list, where each item is a die or a number
    :param index_list: An integer describing which die to take after sorting.
    If it is a list, take the dice in all the places according to the list, and sum the dice
    If it is None, the function will instead return a list of dice, such that
    output[i] = get_pos(dice_list, i)
    :return: The new die
    """

    # If index_list is just a number, turn it into a list of length 1
    if (index_list is not None) and (not isinstance(index_list, list)):
        index_list = [index_list]

    # Turn each value in the dice list, to a trivial die with the same value
    for i in range(len(dice_list)):
        dice_list[i] = dice_utilities.force_cube(dice_list[i])

    # Treat negative numbers as numbers counted from the end of the list
    if index_list is not None:
        index_list = [(index_list[i] % len(dice_list)) for i in range(len(index_list))]

    # Generate all combinations of dice that can be rolled
    all_combs = dice_utilities.generate_all_dice_combs(dice_list)

    # Sort all the combinations
    for i in range(len(all_combs)):
        chance = all_combs[i][0]
        new_list = sorted(all_combs[i][1], reverse=True)
        all_combs[i] = (chance, new_list)

    # If index_list is None, we create a new dice list, which holds the die for each order statistics
    if index_list is None:
        new_dice_list = []
        # We first loop over all the order statistics
        for index in range(len(dice_list)):
            new_dice = dice()
            for i in range(len(all_combs)):
                # For each combination, take the relevant value from the sorted list (to get the order statistic)
                value = all_combs[i][1][index]
                chance = all_combs[i][0]
                # Update the new dices' statistics according to the chances to get each value
                if not (value in new_dice.pdf):
                    new_dice.pdf[value] = 0
                new_dice.pdf[value] += chance
            # Add the new order statistics cube to the list
            new_dice_list.append(new_dice)

        return new_dice_list
    else:
        # If index_list is not None, we only return one die
        new_dice = dice()
        for i in range(len(all_combs)):
            # For each combination, take the values according to the index list, and sum the result
            value = sum([all_combs[i][1][index_list[j]] for j in range(len(index_list))])
            chance = all_combs[i][0]
            # Update the new dice statistics according to the chances to get each value
            if not (value in new_dice.pdf):
                new_dice.pdf[value] = 0
            new_dice.pdf[value] += chance

        return new_dice


def drop_pos(dice_list: list[dice | float | int], index_list: int | list[int]) -> dice:
    """
    Simulates the result of rolling multiple dice, sorting them, removing values after the sorting, and summing the dice
    :param dice_list: A list, where each item is a die or a number
    :param index_list: An integer describing which die to drop after sorting.
    If it is a list, drop the dice in all the places according to the list, and sum the dice
    :return: The new die
    """

    if not isinstance(index_list, list):
        index_list = [index_list]

    # We first perform modulus on each index, so -1 will turn to len(list)-1
    mod_index_list = [(index_list[i] % len(dice_list)) for i in range(len(index_list))]
    # We convert the list of indices to drop, to a list of indices to keep
    keep_index_list = [i for i in range(len(dice_list)) if not (i in mod_index_list)]
    # We call the get_pos method with the list of indices to keep
    return get_pos(dice_list, keep_index_list)


def func(lambda_func: Callable[..., int | float | dice], *args, **kwargs) -> tuple[dice, ...] | Callable[
    ..., int | float | dice]:
    """
    Executes a generic function on dice
    :param lambda_func: The function to execute
    :param args: The dice and values to pass to the function
    :return: A new die, according to the function
    """

    # If args and kwargs are empty, treat this call as a decorator
    if not args and not kwargs:
        def inner(*inner_args, **inner_kwargs):
            return func(lambda_func, *inner_args, **inner_kwargs)

        return inner

    # Turn the tuple to a list
    dice_list = list(args)

    # Turn each value in the dice list, to a trivial die with the same value
    for i in range(len(dice_list)):
        dice_list[i] = dice_utilities.force_cube(dice_list[i])

    # Generate all combinations of the input dice
    all_combs = dice_utilities.generate_all_dice_combs(dice_list)

    # Call the lambda function with the first combination, to see what it returns
    # If it returns a tuple, this method will return a tuple of dice
    # The length of the tuple we return is the same as the length of the tuple the lambda function returns
    # We create a list now, but we will turn it to a tuple later
    if isinstance(lambda_func(*all_combs[0][1]), tuple):
        return_len = len(lambda_func(*all_combs[0][1]))
        new_dice_list = [dice() for _ in range(return_len)]
        for comb in all_combs:
            # For each combination, take the values according to the given function and current combination
            value_list = list(lambda_func(*comb[1]))
            # We generalize, and turn each returned value to a die (if it is not already)
            for i in range(len(value_list)):
                value_list[i] = dice_utilities.force_cube(value_list[i])

            comb_chance = comb[0]

            # Loop over all the output from the function
            # And update the new dice statistics according to the chances to get each value
            for i in range(len(value_list)):
                for roll, chance in value_list[i].pdf.items():
                    if not (roll in new_dice_list[i].pdf):
                        new_dice_list[i].pdf[roll] = 0
                    new_dice_list[i].pdf[roll] += chance * comb_chance

        return tuple(new_dice_list)

    # Now we treat the case where the lambda function returns a single value or die
    else:
        new_dice = dice()
        for comb in all_combs:
            # For each combination, take the values according to the given function
            value = lambda_func(*comb[1], **kwargs)
            value = dice_utilities.force_cube(value)

            comb_chance = comb[0]

            # Update the new dice statistics according to the chances to get each value
            for roll, chance in value.pdf.items():
                if not (roll in new_dice.pdf):
                    new_dice.pdf[roll] = 0
                new_dice.pdf[roll] += chance * comb_chance

        return new_dice


def highest(*args: dice | float | int) -> dice:
    """
    :param args: Dice and values
    :return: A new die, describing taking the highest value of the given dice and values
    """

    if len(args) == 1:
        # If we are given only one argument, and it is a list, treat it as though we received a tuple
        if isinstance(args[0], list):
            args = tuple(args[0])
        # If we are only given one argument, and it is a die or a number, we have reached the recursion stop case
        # The highest value of one argument, is the argument, so we simply return it
        else:
            return dice_utilities.force_cube(args[0])

    # We recursively call the 'highest' function
    # We split the input tuple in 2, and call the 'highest' function on each half
    # We then find the highest between these two dice
    first = highest(*args[:len(args) // 2])
    second = highest(*args[len(args) // 2:])

    new_dice = dice()

    for first_roll, first_chance in first.pdf.items():
        for second_roll, second_chance in second.pdf.items():
            new_roll = max(first_roll, second_roll)
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += first_chance * second_chance

    return new_dice


def lowest(*args: dice | float | int) -> dice:
    """
    :param args: Dice and values
    :return: A new die, describing taking the lowest value of the given dice and values
    """

    if len(args) == 1:
        # If we are given only one argument, and it is a list, treat it as though we received a tuple
        if isinstance(args[0], list):
            args = tuple(args[0])
        # If we are only given one argument, and it is a die or a number, we have reached the recursion stop case
        # The lowest value of one argument, is the argument, so we simply return it
        else:
            return dice_utilities.force_cube(args[0])

    # We recursively call the 'lowest' function
    # We split the input tuple in 2, and call the 'lowest' function on each half
    # We then find the lowest between these two dice
    first = lowest(*args[:len(args) // 2])
    second = lowest(*args[len(args) // 2:])

    new_dice = dice()

    for first_roll, first_chance in first.pdf.items():
        for second_roll, second_chance in second.pdf.items():
            new_roll = min(first_roll, second_roll)
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += first_chance * second_chance

    return new_dice


def chain_compare(*args) -> dice:
    """
    Perform a chain comparison of dice
    :param args: Alternating sequence of dice (or numbers) and signs. Must start with a die and end with a die
    :return: A boolean die describing the result of the chain comparison
    """
    # If only three arguments are given, treat it as a normal comparison:
    if len(args) == 3:
        match args[1]:
            case '>':
                return dice_utilities.force_cube(args[0]) > dice_utilities.force_cube(args[2])
            case '>=':
                return dice_utilities.force_cube(args[0]) >= dice_utilities.force_cube(args[2])
            case '<':
                return dice_utilities.force_cube(args[0]) < dice_utilities.force_cube(args[2])
            case '<=':
                return dice_utilities.force_cube(args[0]) <= dice_utilities.force_cube(args[2])
            case '==':
                return dice_utilities.force_cube(args[0]) == dice_utilities.force_cube(args[2])
            case '!=':
                return dice_utilities.force_cube(args[0]) != dice_utilities.force_cube(args[2])
            # If the sign is not one of these six, raise an error
            case _:
                raise Exception('Invalid comparison operator')

    # Create a dictionary that hold only the cases where the comparison is True
    # The key is the value of the last die
    # The value is the probability
    new_dict = {}

    # Check the first comparison of the chain
    die1 = dice_utilities.force_cube(args[0])
    die2 = dice_utilities.force_cube(args[2])
    # Loop over all combination of the two dice
    for roll1, chance1 in die1.pdf.items():
        for roll2, chance2 in die2.pdf.items():
            # According to the sign, perform the correct comparison
            match args[1]:
                case '>':
                    result = roll1 > roll2
                case '>=':
                    result = roll1 >= roll2
                case '<':
                    result = roll1 < roll2
                case '<=':
                    result = roll1 <= roll2
                case '==':
                    result = roll1 == roll2
                case '!=':
                    result = roll1 != roll2
                # If the sign is not one of these six, raise an error
                case _:
                    raise Exception('Invalid comparison operator')

            # Update the <new_dict> if the result is true
            if result:
                # Add the value of the last die
                if roll2 not in new_dict.keys():
                    new_dict[roll2] = 0
                new_dict[roll2] += chance1 * chance2

    # Split <args> to a list of signs and a list of dice
    sign_list = list(args[3::2])
    dice_list = list(args[4::2])
    # Change each die to a die class if it is not already
    for i in range(len(dice_list)):
        dice_list[i] = dice_utilities.force_cube(dice_list[i])

    # Loop over each sign-die arguments
    for i in range(len(sign_list)):
        # Create a new dictionary that will hold the current results, and save the old one
        old_dict = new_dict
        new_dict = {}
        # Loop over all <old_dict> items,
        # i.e. loop over all comparison results up to this point, and all values of the last die
        for dict_tuple, past_chance in old_dict.items():
            # Loop over all the values of the current die
            for roll, chance in dice_list[i].pdf.items():
                # Perform comparison according to the sign
                match sign_list[i]:
                    case '>':
                        result = dict_tuple > roll
                    case '>=':
                        result = dict_tuple >= roll
                    case '<':
                        result = dict_tuple < roll
                    case '<=':
                        result = dict_tuple <= roll
                    case '==':
                        result = dict_tuple == roll
                    case '!=':
                        result = dict_tuple != roll
                    # If the sign is not one of these six, raise an error
                    case _:
                        raise Exception('Invalid comparison operator')

                # We only update the <new_dict> if the result is true
                if result:
                    if roll not in new_dict.keys():
                        new_dict[roll] = 0
                    new_dict[roll] += past_chance * chance

    # Turn the <new_dict> variable to a die, by disregarding the second element
    # (which describes the value of the last die in the comparison) of each key
    new_dice = dice()
    new_dice.pdf[1] = sum(new_dict.values())
    new_dice.pdf[0] = 1 - new_dice.pdf[1]

    return new_dice


def count(dice_list: list[dice], *args) -> dice:
    """
    Simulates rolling a collection of dice, and checking if it appears in the list.
    The value is based on the amount of times the roll appears in the list
    :param dice_list: The collection of dice
    :param args: A number or a list to compare. Can be a nested list
    :return: A new die
    """
    return sum([die.count(*args) for die in dice_list])


# ~~~~~~~~~ Plotting ~~~~~~~~~


def print_summary(dice_list: list[dice], desired_prints: int | list[int] | None = None) -> None:
    """
    Prints basic information on a collection of dice
    :param dice_list: A list of dice
    :param desired_prints: Specifies what information to print. Format as a list.
                           If the list has any of these value, it will print:
                                0 - Mean
                                1 - Deviation
                                2 - Maximum
                                3 - Minimum
                           By default, the function prints all info
    """

    if desired_prints is None:
        desired_prints = [0, 1, 2, 3]

    if not isinstance(desired_prints, list):
        desired_prints = [desired_prints]

    max_pipes = 150
    filled_char = '█'
    empty_char = ' '
    end_char = '-'

    mean_list = dice_utilities.evaluate_dice_list(lambda x: x.mean(), dice_list)
    std_list = dice_utilities.evaluate_dice_list(lambda x: x.std(), dice_list)
    max_list = dice_utilities.evaluate_dice_list(lambda x: x.max(), dice_list)
    min_list = dice_utilities.evaluate_dice_list(lambda x: x.min(), dice_list)

    lists_list = [mean_list, std_list, max_list, min_list]
    string_list = ['Mean', 'Deviation', 'Maximum', 'Minimum']

    max_val = max(max_list)

    for j in desired_prints:
        header_text = ['Output', '#', string_list[j]]
        text = [header_text]
        for i in range(len(lists_list[j])):
            num_pipes = round(max_pipes * lists_list[j][i] / max_val)
            progress_bar_string = filled_char * num_pipes + empty_char * (max_pipes - num_pipes) + end_char
            text += [[dice_list[i].name, '{:.2f}'.format(lists_list[j][i]), progress_bar_string]]
        print(tabulate(text, headers='firstrow') + '\n')


def plot_stats(dice_list: list[dice | Callable[..., dice]], dynamic_vars=None, x_label: str = 'Value',
               title: str | None = None, y_lim=None) -> None:
    """
    Plot basic stats (normal [pdf], atmost [cdf] and atleast [1+pdf-cdf]) of a list of dice
    :param y_lim: Defines the y-limits of the plot.
                    Set as `None` for auto y-lim.
                    Set as a 2-length tuple or list to explicitly define the y-limits (min, max).
                    Set as `'max'` so the plot will scale will the graph but will only become bigger
    :param dice_list: A list of dice to plot the info of
    :param dynamic_vars: A list of lists. Each list describes a slider in the plot, in the following format:
                                            - Slider name
                                            - Slider minimum value
                                            - Slider initial value
                                            - Slider maximum value
                                            - Slider step size (default 1)
                        If dynamic variables are given, the dice list needs to be a list of functions.
                        These functions take a number of arguments equal to the number of sliders (in the appropriate order)
                        And they need to return a dice object
    :param x_label: Optional parameter, the x-label of the plot
    :param title: Optional parameter, the title of the plot
    """
    d_fig, d_ax = plt.subplots()
    if dynamic_vars is not None:
        i_fig = plt.figure()

    if dynamic_vars is None:
        dynamic_vars = []
    elif not isinstance(dynamic_vars[0], list):
        dynamic_vars = [dynamic_vars]

    # Set default step size
    for dynamic_var in dynamic_vars:
        if len(dynamic_var) == 4:
            dynamic_var.append(1)
        if len(dynamic_var) != 5 or not (
                isinstance(dynamic_var[0], str) and isinstance(dynamic_var[1], (float, int)) and
                isinstance(dynamic_var[2], (float, int)) and isinstance(dynamic_var[3], (float, int)) and
                isinstance(dynamic_var[4], (float, int))):
            raise TypeError("dynamic_var must be of the form "
                            "[name, minimal value, starting value, maximal value, <step size>]")

    # If only one die was given, and not as a list, we turn it into a list of length 1
    if not isinstance(dice_list, list):
        dice_list = [dice_list]

    # Add a field to d_ax that holds the maximum values of the y-lim
    d_ax.max_y_lim = None

    # Create the list of sliders according to the list of parameters given
    sliders_list = []
    slider_height = 0.05
    slider_spacing = 0.00
    total_slider_size = len(dynamic_vars) * (slider_height + slider_spacing) + 0.1
    for i in range(len(dynamic_vars)):
        dynamic_var = dynamic_vars[i]
        slider_loc = slider_spacing * i + slider_height * i
        ax_sld = plt.axes([0.3, slider_loc, 0.5, slider_height])
        slider = Slider(ax=ax_sld,
                        label=dynamic_var[0],
                        valmin=dynamic_var[1],
                        valinit=dynamic_var[2],
                        valmax=dynamic_var[3],
                        valstep=dynamic_var[4],
                        initcolor='none')
        sliders_list.append(slider)

    # Define the function data, which gets the current mode (normal, atleast, atmost).
    # The function gets the values of the sliders, and calls each die function to generate a list of dice
    # It then calls the get_data_for_plot function, which turns it into a list of floats

    # There aren't any dynamic variables - treat the dice as dice
    if not dynamic_vars:
        def data(mode):
            plot_kwargs = {}
            xy_data = dice_utilities.plot_stats__get_data_for_plot(dice_list, mode[0])
            plot_kwargs['x_data'] = [xy_data_elem[0] for xy_data_elem in xy_data]
            plot_kwargs['y_data'] = [xy_data_elem[1] for xy_data_elem in xy_data]
            plot_kwargs['legend_labels'] = [xy_data_elem[2] for xy_data_elem in xy_data]
            plot_kwargs['x_label'] = x_label
            plot_kwargs['y_lim'] = y_lim
            match mode[0]:
                case 'normal':
                    plot_kwargs['y_label'] = 'Probability - Normal'
                case 'atmost':
                    plot_kwargs['y_label'] = 'Probability - At Most'
                case 'atleast':
                    plot_kwargs['y_label'] = 'Probability - At Least'
                case _:
                    plot_kwargs['y_label'] = 'Probability'
            return plot_kwargs
    # There are dynamic variables - treat the dice as functions
    else:
        def data(mode):
            plot_kwargs = {}
            xy_data = dice_utilities.plot_stats__get_data_for_plot(
                [die(*dice_utilities.extract_values_from_sliders(sliders_list)) for die in dice_list],
                mode[0])
            plot_kwargs['x_data'] = [xy_data_elem[0] for xy_data_elem in xy_data]
            plot_kwargs['y_data'] = [xy_data_elem[1] for xy_data_elem in xy_data]
            plot_kwargs['legend_labels'] = [xy_data_elem[2] for xy_data_elem in xy_data]
            plot_kwargs['x_label'] = dice_utilities.plot_stats__get_x_label(x_label, sliders_list)
            plot_kwargs['y_lim'] = y_lim
            match mode[0]:
                case 'normal':
                    plot_kwargs['y_label'] = 'Probability - Normal'
                case 'atmost':
                    plot_kwargs['y_label'] = 'Probability - At Most'
                case 'atleast':
                    plot_kwargs['y_label'] = 'Probability - At Least'
                case _:
                    plot_kwargs['y_label'] = 'Probability'
            return plot_kwargs

    # The current mode that is shown. It is a list so it can be mutable
    chosen_mode = ['normal']
    # Attach the function of updating the plot when the slider changes
    for slider in sliders_list:
        slider.on_changed(lambda event:
                          dice_utilities.update_plot(d_fig, d_ax, lines_list, data(chosen_mode)))
        # slider.on_changed(lambda event: print(chosen_mode))

    # Create the list of line graphs for the different dice
    lines_list = []
    for i in range(len(dice_list)):
        # The plot is empty at first, but it will be populated later
        line, = d_ax.plot([], [], 'o-', linewidth=3, label='')
        lines_list.append(line)

    # Info about the buttons locations and sizes
    button_sz_x = 2 / 14
    button_sz_y = 1 / 21
    button_spacing_x = 1 / 14
    button_margin_x = (1 - (3 * button_sz_x) - (2 * button_spacing_x)) / 2
    button_margin_y = 1 / 21
    hovercolor = '0.575'

    # Create the three buttons
    # For each button we attach an event that occurs on click. In it, we update the 'global' variable chosen_mode
    # We then update the plot according to the new mode

    # Button - Normal
    normal_ax = plt.axes(
        [button_margin_x, 1 - button_margin_y, button_sz_x, button_sz_y])
    normal_button = Button(normal_ax, 'Normal', hovercolor=hovercolor)

    def normal_button_click(event):
        chosen_mode[0] = 'normal'
        dice_utilities.update_plot(d_fig, d_ax, lines_list, data(chosen_mode))

    normal_button.on_clicked(normal_button_click)

    # Button - At Least
    atleast_ax = plt.axes(
        [button_margin_x + (button_sz_x + button_spacing_x), 1 - button_margin_y, button_sz_x, button_sz_y])
    atleast_button = Button(atleast_ax, 'At Least', hovercolor=hovercolor)

    def atleast_button_click(event):
        chosen_mode[0] = 'atleast'
        dice_utilities.update_plot(d_fig, d_ax, lines_list, data(chosen_mode))

    atleast_button.on_clicked(atleast_button_click)

    # Button - At Most
    atmost_ax = plt.axes(
        [button_margin_x + (button_sz_x + button_spacing_x) * 2, 1 - button_margin_y, button_sz_x, button_sz_y])
    atmost_button = Button(atmost_ax, 'At Most', hovercolor=hovercolor)

    def atmost_button_click(event):
        chosen_mode[0] = 'atmost'
        dice_utilities.update_plot(d_fig, d_ax, lines_list, data(chosen_mode))

    atmost_button.on_clicked(atmost_button_click)

    # Simulate clicking on the 'normal' button to create the first plot
    normal_button_click(None)

    # General plot details
    # d_ax.set_xlabel(x_label)
    # d_ax.set_ylabel('Probability')
    if title is not None:
        d_ax.set_title(title)
    # d_ax.legend()
    d_ax.grid(alpha=0.3)
    plt.show()


def plot_mean(dice_list: list[dice | Callable[..., dice]], dynamic_vars, title: str | None = None, y_lim=None) -> None:
    """
    Plot the mean of a list of dice
    :param y_lim: Defines the y-limits of the plot.
                    Set as `None` for auto y-lim.
                    Set as a 2-length tuple or list to explicitly define the y-limits (min, max).
                    Set as `'max'` so the plot will scale will the graph but will only become bigger
    :param dice_list: A list of dice to plot the info of
    :param dynamic_vars: A list of lists. Each list describes a slider in the plot, in the following format:
                                            - Slider name
                                            - Slider minimum value
                                            - Slider initial value
                                            - Slider maximum value
                                            - Slider step size (default 1)
                        If dynamic variables are given, the dice list needs to be a list of functions.
                        These functions take a number of arguments equal to the number of sliders (in the appropriate order)
                        And they need to return a dice object
    :param title: Optional parameter, the title of the plot
    """
    d_fig, d_ax = plt.subplots()
    i_fig = plt.figure()

    # If only one die is given, turn it to a list
    if not isinstance(dice_list, list):
        dice_list = [dice_list]

    # If only one dynamic var is given, and it is not wrapped correctly, wrap it in another list
    if not isinstance(dynamic_vars[0], list):
        dynamic_vars = [dynamic_vars]

    # Set default step size
    for dynamic_var in dynamic_vars:
        if len(dynamic_var) == 4:
            dynamic_var.append(1)
        if len(dynamic_var) != 5 or not (
                isinstance(dynamic_var[0], str) and isinstance(dynamic_var[1], (float, int)) and
                isinstance(dynamic_var[2], (float, int)) and isinstance(dynamic_var[3], (float, int)) and
                isinstance(dynamic_var[4], (float, int))):
            raise TypeError("dynamic_var must be of the form "
                            "[name, minimal value, starting value, maximal value, <step size>]")

    # Add a field to d_ax that holds the maximum values of the y-lim
    d_ax.max_y_lim = None

    # Define the lists that contain all the buttons and sliders
    # We define them now because we need them for the update function
    sliders_list = []
    buttons_list = []

    # Define the update function. Which updates the plot according to the sliders and buttons
    def update():
        plot_kwargs = dice_utilities.plot_mean__get_data_by_slider(dice_list,
                                                                   sliders_list,
                                                                   buttons_list)
        plot_kwargs['y_lim'] = y_lim
        dice_utilities.update_plot(d_fig, d_ax, lines_list, plot_kwargs)

    # Parameters for the gui
    pgui_wbutton = 0.2  # Width of the button
    pgui_wslider = 0.6  # Width of the slider
    pgui_wmargin = (1 - pgui_wbutton - pgui_wslider) / 3  # Margin in the x-axis between all components
    pgui_hmargin = 0.02  # Margin in the y-axis between each button-slider pair
    pgui_height = 0.05  # Height of the button and the slider
    pgui_hovercolor = '0.575'  # Color of the button on hover

    # Create the list of sliders according to the list of parameters given
    for i in range(len(dynamic_vars)):
        dynamic_var = dynamic_vars[i]
        # Calculate this slider's distance from the bottom of the page
        slider_loc = pgui_height * i + pgui_hmargin * (i + 1)
        # Set the slider's location and size
        ax_sld = plt.axes([2 * pgui_wmargin + pgui_wbutton, slider_loc, pgui_wslider, pgui_height])
        # Set the slider's minimal, initial, and maximal value, as well as the step size
        slider = Slider(ax=ax_sld,
                        label='',
                        valmin=dynamic_var[1],
                        valinit=dynamic_var[2],
                        valmax=dynamic_var[3],
                        valstep=dynamic_var[4],
                        initcolor='none')
        # Attach the function of updating the plot when the slider changes
        slider.on_changed(lambda event: update())
        # Push the slider to the list of sliders
        sliders_list.append(slider)

    for i in range(len(dynamic_vars)):
        dynamic_var = dynamic_vars[i]
        # Calculate this button's distance from the bottom of the page
        button_loc = pgui_height * i + pgui_hmargin * (i + 1)
        # Set the button's location and size
        ax_btn = plt.axes([pgui_wmargin, button_loc, pgui_wbutton, pgui_height])
        # Create the button
        button = Button(ax=ax_btn,
                        label=dynamic_var[0],
                        hovercolor=pgui_hovercolor)
        # Set the buttons on click event
        button.on_clicked(dice_utilities.plot_mean__set_button_callback(buttons_list, button, update))
        # Push the button to the list of buttons
        buttons_list.append(button)
    # Set the first button to be the one that is initially pushed. We mark a pushed button by setting it inactive
    buttons_list[0].active = False

    # Create a function that creates a data structure for the plot

    # Create the list of line graphs for the different dice
    lines_list = []
    for i in range(len(dice_list)):
        # The plot is empty at first, but it will be populated later
        line, = d_ax.plot(np.array([]), np.array([]), 'o-', linewidth=3, label='')
        lines_list.append(line)

    update()
    # General plot details
    d_ax.set_ylabel('Mean')
    if title is not None:
        d_ax.set_title(title)
    d_ax.grid(alpha=0.3)
    plt.show()
