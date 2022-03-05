import random
import copy
import math
import dice_utilities
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from matplotlib.widgets import Slider, Button


class dice:
    def __init__(self):
        self.pdf = {}
        self.name = "Unnamed Die"

    def print_normal(self, name=None):
        """
        Prints general info about the dice (mean and std), and prints a graph-like plot of the pdf of the dice
        :param name: Optional Argument, if given, sets the dice name according to this argument
        """

        dice_utilities.print_data(self, self.pdf, name)

    def print_at_least(self, name=None):
        """
        Prints general info about the dice (mean and std), and prints a graph-like plot about the chance to get at least a value
        :param name: Optional Argument, if given, sets the dice name according to this argument
        """

        dice_utilities.print_data(self, self.at_least(), name)

    def print_at_most(self, name=None):
        """
        Prints general info about the dice (mean and std), and prints a graph-like plot about the chance to get at most a value
        :param name: Optional Argument, if given, sets the dice name according to this argument
        """

        dice_utilities.print_data(self, self.at_most(), name)

    def at_least(self):
        """
        :return: A dictionary describing the chance to get at least that result
        """
        new_dict = {}
        old_value = 1
        for roll, chance in sorted(self.pdf.items()):
            new_dict[roll] = old_value
            old_value -= self.pdf[roll]
        return new_dict

    def at_most(self):
        """
        :return: A dictionary describing the chance to get at most that result
        """
        new_dict = {}
        old_value = 0
        for roll, chance in sorted(self.pdf.items()):
            old_value += self.pdf[roll]
            new_dict[roll] = old_value
        return new_dict

    def mean(self):
        """
        :return: Returns the mean of the dice
        """
        res = 0
        for roll, chance in sorted(self.pdf.items()):
            res += roll * chance
        return res

    def std(self):
        """
        :return: Returns the standard deviation of the dice
        """
        m = self.mean()
        var = 0
        for roll, chance in sorted(self.pdf.items()):
            var += ((roll - m) ** 2) * chance
        return math.sqrt(var)

    def max(self):
        """
        :return: The maximum value the die can roll
        """
        return max(self.pdf.keys())

    def min(self):
        """
        :return: The minimum value the die can roll
        """
        return min(self.pdf.keys())

    def roll(self, n=0):
        """
        Simulates a roll of the dice
        :param n: How many dice rolls to simulate
        :return: The simulates roll(s). If a value was given to n, returns a list, otherwise returns a value
        """
        rolls = list(self.pdf.keys())
        chances = list(self.pdf.values())
        nFix = n if n >= 1 else 1
        index = random.choices(range(len(chances)), chances, k=nFix)
        if n == 0:
            return rolls[index[0]]
        return [rolls[index[i]] for i in range(len(index))]

    def norm(self):
        """
        Debugging method
        :return: The sum of all probabilities of the die. Should be 1 at all times
        """
        return sum(self.pdf.values())

    # ~~~~~~~~~ Overloaded Binary and Unary Operations ~~~~~~~~~

    def __add__(self, other):
        """
        Creates a die describing the sum of two dice
        :param other: The second die to add. Can be a number
        :return: A new die, with statistics according to the sum of the two dice
        """
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = self_rolls[i] + other_rolls[j]
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __sub__(self, other):
        """
        Creates a die describing the difference of two dice
        :param other: The second die to subtract. Can be a number
        :return: A new die, with statistics according to the difference of the two dice
        """
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = self_rolls[i] - other_rolls[j]
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __mul__(self, other):
        """
        Creates a die describing the product of a die with a number
        :param other: The number to multiply by
        :return: A new die, with statistics according to the product of the die and the number
        """
        new_dice = dice()
        for roll, chance in self.pdf.items():
            new_roll = roll * other
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[roll * other] += chance
        return new_dice

    def __truediv__(self, other):
        """
        Creates a die describing the quotient of a die with a number. Division is done in floating point calculation
        :param other: The number to divide by
        :return: A new die, with statistics according to the quotient of the die and the number
        """
        new_dice = dice()
        for roll, chance in self.pdf.items():
            new_dice.pdf[roll / other] = chance
        return new_dice

    def __mod__(self, other):
        """
        Creates a die describing the modulus of a die with a number. Division is done in floating point calculation
        :param other: The number to divide by
        :return: A new die, with statistics according to the modulus of the die and the number
        """
        new_dice = dice()
        for roll, chance in self.pdf.items():
            new_dice.pdf[roll % other] = chance
        return new_dice

    def __floordiv__(self, other):
        """
        Creates a die describing the quotient of a die with a number. Division is done in integer calculation
        :param other: The number to divide by
        :return: A new die, with statistics according to the quotient of the die and the number
        """
        new_dice = dice()
        for roll, chance in self.pdf.items():
            new_roll = roll // other
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += chance
        return new_dice

    def __pow__(self, power, modulo=None):
        """
        Creates a die describing rolling the same die multiple times and taking the sum
        :param power: How many dice to roll
        :return: A new die, with statistics according to the sum of the die
        """
        if power == 1:
            return copy.deepcopy(self)
        else:
            return (self ** (power - 1)) + copy.deepcopy(self)

    def __neg__(self):
        """
        :return: Flips the sign of the values of the dice
        """
        new_dice = dice()
        for roll, chance in self.pdf.items():
            new_dice.pdf[-roll] = chance
        return new_dice

    # ~~~~~~~~~ Common die manipulation ~~~~~~~~~

    def adv(self, n=2):
        """
        :return: A new die, describing rolling the current die twice, and taking the higher result
        """
        return highest(*((self,) * n))

    def dis(self, n=2):
        """
        :return: A new die, describing rolling the current die twice, and taking the lower result
        """
        return lowest(*((self,) * n))

    def exp(self, explode_on=None, max_depth=2):
        """
        Simulates exploding a die, i.e., rolling the die, and if it rolls the highest value, roll it again
        :param explode_on: Optional parameter. If given, the die will explode on this value, instead of the maximum value of the die
        :param max_depth: Maximum depth of the simulation. Default is 2
        :return: A new die
        """

        if explode_on is None:
            explode_on = self.max()

        if max_depth == 0:
            return self

        new_dice = dice()
        deeper_dice = self.exp(max_depth=max_depth - 1)

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        deeper_dice_rolls = list(deeper_dice.pdf.keys())
        deeper_dice_chances = list(deeper_dice.pdf.values())

        new_dice = dice()

        for i in range(len(self_rolls)):
            for j in range(len(deeper_dice_rolls)):
                if self_rolls[i] == explode_on:
                    new_roll = self_rolls[i] + deeper_dice_rolls[j]
                else:
                    new_roll = self_rolls[i]
                if not (new_roll in new_dice.pdf):
                    new_dice.pdf[new_roll] = 0
                new_dice.pdf[new_roll] += self_chances[i] * deeper_dice_chances[j]

        return new_dice

    def count(self, *args):
        """
        Simulates rolling a die, and checking if it appears in the list.
        The value is based on the amount of times the roll appears in the list
        :param args: A number or a list to compare. Can be a nested list
        :return: A new die
        """

        match_list = list(args)

        if not isinstance(match_list, list):
            match_list = [match_list]

        # flat = lambda S: S if S == [] else (flat(S[0]) + flat(S[1:]) if isinstance(S[0], list) else S[:1] + flat(S[1:]))
        flattened_match_list = dice_utilities.flatten(match_list)

        new_dice = dice()
        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())

        for i in range(len(self_rolls)):
            new_roll = flattened_match_list.count(self_rolls[i])
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += self_chances[i]

        return new_dice

    def cond(self, die_if_true, die_if_false):
        """
        Uses the first boolean die to choose a second die
        :param die_if_true: The die to choose if the first one is True
        :param die_if_false: The die to choose if the first one is False
        :return: The new die
        """

        if not isinstance(die_if_true, dice):
            die_if_true = from_const(die_if_true)
        if not isinstance(die_if_false, dice):
            die_if_false = from_const(die_if_false)

        new_dice = dice()

        for roll, chance in die_if_false.pdf.items():
            new_dice.pdf[roll] = chance * self.pdf[0]

        for roll, chance in die_if_true.pdf.items():
            if roll not in new_dice.pdf.keys():
                new_dice.pdf[roll] = 0
            new_dice.pdf[roll] += chance * self.pdf[1]

        return new_dice

    def switch(self, *args):
        """
        Uses the first die (which rolls integers between 0 and N-1) to choose a second die
        :param args: A list or tuple of the other set of dice
        :return: The new die
        """

        if isinstance(args[0], list):
            other_dice_list = list(args[0])
        else:
            other_dice_list = list(args)

        for i in range(len(other_dice_list)):
            if not isinstance(other_dice_list[i], dice):
                other_dice_list[i] = from_const(other_dice_list[i])

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other_dice_list[self_roll].pdf.items():
                if other_roll not in new_dice.pdf.keys():
                    new_dice.pdf[other_roll] = 0
                new_dice.pdf[other_roll] += self_chance * other_chance

        return new_dice

    def dictionary(self, roll_dictionary, default_value=None):
        """
        Transforms the die according to a dictionary
        :param roll_dictionary: A dictionary describing the transformation, has <float> input and <die>/<float> output
        :param default_value: Optional parameter. If given, values that do not have a dictionary value will get this value instead
        :return: The new die
        """

        new_dice = dice()
        # Loop over all the values the self die can roll
        for roll, chance in self.pdf.items():
            # Check if the current roll is an entry in the dictionary
            # If it is, use the dictionary. If not, use the default value or raise an error
            if roll in roll_dictionary.keys():
                new_value = roll_dictionary[roll]
                # Transform the new roll into a die if it is not already
                if not isinstance(new_value, dice):
                    new_value = from_const(new_value)
                for new_roll, new_chance in new_value.pdf.items():
                    if not (new_roll in new_dice.pdf):
                        new_dice.pdf[new_roll] = 0
                    new_dice.pdf[new_roll] += chance * new_chance
            else:
                if default_value is None:
                    raise()
                else:
                    new_value = default_value
                    # Transform the new roll into a die if it is not already
                    if not isinstance(new_value, dice):
                        new_value = from_const(new_value)
                    for new_roll, new_chance in new_value.pdf.items():
                        if not (new_roll in new_dice.pdf):
                            new_dice.pdf[new_roll] = 0

        return new_dice

    # ~~~~~~~~~ Overloaded Comparative Operations ~~~~~~~~~

    def __lt__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = (self_rolls[i] < other_rolls[j]) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __le__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = (self_rolls[i] <= other_rolls[j]) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __eq__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = (self_rolls[i] == other_rolls[j]) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __ne__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = (self_rolls[i] != other_rolls[j]) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __gt__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = (self_rolls[i] > other_rolls[j]) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __ge__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                new_roll = (self_rolls[i] >= other_rolls[j]) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    # ~~~~~~~~~ Overloaded Boolean Operators ~~~~~~~~~

    def __and__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                self_roll_bool = self_rolls[i] != 0
                other_roll_bool = other_rolls[j] != 0
                new_roll = (self_roll_bool and other_roll_bool) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __or__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                self_roll_bool = self_rolls[i] != 0
                other_roll_bool = other_rolls[j] != 0
                new_roll = (self_roll_bool or other_roll_bool) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __xor__(self, other):
        if not isinstance(other, dice):
            new_other = dice()
            new_other.pdf[other] = 1
            other = new_other

        self_rolls = list(self.pdf.keys())
        self_chances = list(self.pdf.values())
        other_rolls = list(other.pdf.keys())
        other_chances = list(other.pdf.values())

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0

        for i in range(len(self_rolls)):
            for j in range(len(other_rolls)):
                self_roll_bool = self_rolls[i] != 0
                other_roll_bool = other_rolls[j] != 0
                new_roll = (self_roll_bool != other_roll_bool) + 0
                new_dice.pdf[new_roll] += self_chances[i] * other_chances[j]

        return new_dice

    def __invert__(self):

        new_dice = dice()
        new_dice.pdf[0] = 0
        new_dice.pdf[1] = 0
        for roll, chance in self.pdf.items():
            new_roll = 1 if roll == 0 else 0
            new_dice.pdf[new_roll] = chance
        return new_dice

    # ~~~~~~~~~ Reroll Methods ~~~~~~~~~

    def reroll_func(self, lambda_cond, max_depth=1):
        """
        A general reroll function. Simulates rolling a die, and rerolling the dice if a certain condition is met
        :param lambda_cond: A lambda function that takes a number and returns a boolean.
        :param max_depth: A parameter describing if the dice is rerolled multiple times if a bad result is continuously rolled
        Can be an integer (1 for only one reroll), or 'inf' for an infinite number of rerolls
        :return: A new die
        """

        new_dice = dice()

        if max_depth == 'inf':
            self_rolls = list(self.pdf.keys())
            self_chances = list(self.pdf.values())
            total_prob = sum([self_chances[i] for i in range(len(self_rolls)) if not lambda_cond(self_rolls[i])])

            for i in range(len(self_rolls)):
                if not lambda_cond(self_rolls[i]):
                    new_dice.pdf[self_rolls[i]] = self_chances[i] / total_prob

        else:
            if max_depth == 0:
                return self

            deeper_dice = self.reroll_func(lambda_cond, max_depth=max_depth - 1)

            self_rolls = list(self.pdf.keys())
            self_chances = list(self.pdf.values())
            deeper_dice_rolls = list(deeper_dice.pdf.keys())
            deeper_dice_chances = list(deeper_dice.pdf.values())

            new_dice = dice()

            for i in range(len(self_rolls)):
                for j in range(len(deeper_dice_rolls)):
                    if lambda_cond(self_rolls[i]):
                        new_roll = deeper_dice_rolls[j]
                    else:
                        new_roll = self_rolls[i]
                    if not (new_roll in new_dice.pdf):
                        new_dice.pdf[new_roll] = 0
                    new_dice.pdf[new_roll] += self_chances[i] * deeper_dice_chances[j]

        return new_dice

    def reroll_on(self, match_list, max_depth=1):
        """
        Simulates rolling a dice, and rerolling the dice if the value is in match_list
        :param match_list: A number or a list to compare. Can be a nested list
        :param max_depth: A parameter describing if the dice is rerolled multiple times if a bad result is continuously rolled
        Can be an integer (1 for only one reroll), or 'inf' for an infinite number of rerolls
        :return: A new die
        """

        if not isinstance(match_list, list):
            match_list = [match_list]

        # flat = lambda S: S if S == [] else (flat(S[0]) + flat(S[1:]) if isinstance(S[0], list) else S[:1] + flat(S[1:]))
        flattened_match_list = dice_utilities.flatten(match_list)

        return self.reroll_func(lambda x: x in flattened_match_list, max_depth)


# ~~~~~~~~~ Custom Dice Constructors ~~~~~~~~~

def d(N):
    """
    Constructs a new die, which is uniformly distributed over 1 .. N
    :param N: The size of the die
    :return: The new die
    """
    new_dice = dice()
    for i in range(N):
        new_dice.pdf[i + 1] = 1.0 / N
    return new_dice


def standard_dice():
    """
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    :return: The basic 7 dice
    """
    d4 = d(4)
    d6 = d(6)
    d8 = d(8)
    d10 = d(10)
    d12 = d(12)
    d20 = d(20)
    d100 = d(100)

    return d4, d6, d8, d10, d12, d20, d100


def from_const(k):
    """
    Constructs a trivial die with only one value
    :param k: The value of the die
    :return: The new die
    """
    new_dice = dice()
    new_dice.pdf[k] = 1
    return new_dice


def zero():
    """
    Constructs a trivial die with only value of 0
    :return: The new die
    """
    new_dice = dice()
    new_dice.pdf[0] = 1
    return new_dice


def list2dice(values_list):
    """
    Constructs a new die according to a list, such that the chance to get each value in the list is equiprobable
    If a value appears multiple times in the list, the chance of rolling it increases accordingly
    :param values_list: The list describing the new dice
    :return: The new die
    """
    new_dice = dice()
    for elem in values_list:
        if not (elem in new_dice.pdf):
            new_dice.pdf[elem] = 0
        new_dice.pdf[elem] += 1 / len(values_list)

    return new_dice

    # ~~~~~~~~~ Dice Functions ~~~~~~~~~


def range2dice(*args):
    """
    Creates a die with the possible results of 0 .. N-1 with probability according to the input
    :param args: The weights of the different values
    :return: The new die
    """

    if isinstance(args[0], list):
        chance_list = args[0]
    else:
        chance_list = list(args)

    norm_factor = sum(chance_list)
    chance_list_norm = [chance / norm_factor for chance in chance_list]

    new_dice = dice()
    for i in range(len(chance_list_norm)):
        new_dice.pdf[i] = chance_list_norm[i]

    return new_dice


def binary(chance):
    """
    Creates a die with results of 0 or 1
    :param chance: The chance to get 1
    :return: The new die
    """
    new_dice = dice()
    new_dice.pdf[0] = 1 - chance
    new_dice.pdf[1] = chance
    return new_dice


# ~~~~~~~~~ Other Dice Methods ~~~~~~~~~


def get_pos(dice_list, index_list):
    """
    Simulates the result of rolling multiple dice, sorting them, taking values after the sorting, and summing the dice
    :param dice_list: A list, where each item is a die or a number
    :param index_list: An integer describing which die to take after sorting.
    If it is a list, take the dice in all the places according to the list, and sum the dice
    :return: The new die
    """

    # If index_list is just a number, turn it into a list of length 1
    if not isinstance(index_list, list):
        index_list = [index_list]

    # Turn each value in the dice list, to a trivial die with the same value
    for i in range(len(dice_list)):
        if not isinstance(dice_list[i], dice):
            dice_list[i] = from_const(dice_list[i])

    # Treat negative numbers as numbers counted from the end of the list
    mod_index_list = [(index_list[i] % len(dice_list)) for i in range(len(index_list))]

    new_dice = dice()
    # all_combs is a list of tuples.
    # The first element of the tuple describes the probability of this outcome.
    # The second element of the tuple is the corresponding output (i.e. the list of values)
    all_combs = [(1, [])]
    # We first loop over each die, recursively updating the list of all combinations
    for each_dice in dice_list:
        old_combs = copy.deepcopy(all_combs)
        all_combs = []
        # We then loop over all the combinations we have from the previous dice
        for i in range(len(old_combs)):
            # And finally, we loop over all the values of the new die
            for roll, chance in each_dice.pdf.items():
                # The chance to get this new outcome, is the chance to get the list of values of the older dice,
                # times the chance to get the value of the current dice
                new_chance = old_combs[i][0] * chance
                # This outcome is described by a list, of the old values, concatenated with the new value
                new_list = old_combs[i][1] + [roll]
                # Add the new chance-outcome tuple to the list of all combinations
                all_combs += [(new_chance, new_list)]

    # Sort all the combinations
    for i in range(len(all_combs)):
        chance = all_combs[i][0]
        new_list = sorted(all_combs[i][1], reverse=True)
        all_combs[i] = (chance, new_list)

    for i in range(len(all_combs)):
        # For each combination, take the values according to the index list, and sum the result
        value = sum([all_combs[i][1][mod_index_list[j]] for j in range(len(mod_index_list))])
        chance = all_combs[i][0]
        # Update the new dice statistics according to the chances to get each value
        if not (value in new_dice.pdf):
            new_dice.pdf[value] = 0
        new_dice.pdf[value] += chance

    return new_dice


def drop_pos(dice_list, index_list):
    """
    Simulates the result of rolling multiple dice, sorting them, removing values after the sorting, and summing the dice
    :param dice_list: A list, where each item is a die or a number
    :param index_list: An integer describing which die to drop after sorting.
    If it is a list, drop the dice in all the places according to the list, and sum the dice
    :return: The new die
    """

    if not isinstance(index_list, list):
        index_list = [index_list]

    mod_index_list = [(index_list[i] % len(dice_list)) for i in range(len(index_list))]
    keep_index_list = [i for i in range(len(dice_list)) if not (i in mod_index_list)]
    return get_pos(dice_list, keep_index_list)


def func(lambda_func, *args):
    """
    Executes a generic function on dice
    :param lambda_func: The function to execute
    :param args: The dice and values to pass to the function
    :return: A new die, according to the function
    """

    dice_list = list(args)

    # Turn each value in the dice list, to a trivial die with the same value
    for i in range(len(dice_list)):
        if not isinstance(dice_list[i], dice):
            dice_list[i] = from_const(dice_list[i])

    new_dice = dice()
    # all_combs is a list of tuples.
    # The first element of the tuple describes the probability of this outcome.
    # The second element of the tuple is the corresponding output (i.e. the list of values)
    all_combs = [(1, [])]
    # We first loop over each die, recursively updating the list of all combinations
    for each_dice in dice_list:
        old_combs = copy.deepcopy(all_combs)
        all_combs = []
        # We then loop over all the combinations we have from the previous dice
        for i in range(len(old_combs)):
            # And finally, we loop over all the values of the new die
            for roll, chance in each_dice.pdf.items():
                # The chance to get this new outcome, is the chance to get the list of values of the older dice,
                # times the chance to get the value of the current dice
                new_chance = old_combs[i][0] * chance
                # This outcome is described by a list, of the old values, concatenated with the new value
                new_list = old_combs[i][1] + [roll]
                # Add the new chance-outcome tuple to the list of all combinations
                all_combs += [(new_chance, new_list)]

    for i in range(len(all_combs)):
        # For each combination, take the values according to the given function
        value = lambda_func(*all_combs[i][1])
        if not isinstance(value, dice):
            value = from_const(value)

        comb_chance = all_combs[i][0]
        # Update the new dice statistics according to the chances to get each value
        for roll, chance in value.pdf.items():
            if not (roll in new_dice.pdf):
                new_dice.pdf[roll] = 0
            new_dice.pdf[roll] += chance * comb_chance

    return new_dice


def highest(*args):
    """
    :param args: Dice and values
    :return: A new die, describing taking the highest value of the given dice and values
    """

    if len(args) == 1:
        return args[0]
    first = args[0]
    other = highest(*args[1:])

    first_rolls = list(first.pdf.keys())
    first_chances = list(first.pdf.values())
    other_rolls = list(other.pdf.keys())
    other_chances = list(other.pdf.values())

    new_dice = dice()

    for i in range(len(first_rolls)):
        for j in range(len(other_rolls)):
            new_roll = max(first_rolls[i], other_rolls[j])
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += first_chances[i] * other_chances[j]

    return new_dice


def lowest(*args):
    """
    :param args: Dice and values
    :return: A new die, describing taking the lowest value of the given dice and values
    """
    if len(args) == 1:
        return args[0]
    first = args[0]
    other = lowest(*args[1:])

    first_rolls = list(first.pdf.keys())
    first_chances = list(first.pdf.values())
    other_rolls = list(other.pdf.keys())
    other_chances = list(other.pdf.values())

    new_dice = dice()

    for i in range(len(first_rolls)):
        for j in range(len(other_rolls)):
            new_roll = min(first_rolls[i], other_rolls[j])
            if not (new_roll in new_dice.pdf):
                new_dice.pdf[new_roll] = 0
            new_dice.pdf[new_roll] += first_chances[i] * other_chances[j]

    return new_dice


# ~~~~~~~~~ Plotting ~~~~~~~~~


def plot(dice_list, names=None, title=None, dynamic_vars=[]):
    """
    Plots statistics of dice
    :param dice_list: A dice object, or a list of dice objects to plot the statistics of
    :param names: A string or list of strings. If given, override the current names of the dice
    :param title: Optional title for the plot, replaces the default one
    :param dynamic_vars: A list of tuples, describing dynamic variables of the plot.
    If this list is not empty, dice_list should instead be a list of lambda functions which generate the dice
    The tuple is in the form: <name>, <min_val>, <start_val>, <max_val>, <val_step>
    """

    # If only one die was given, and not as a list, we turn it into a list of length 1
    if not isinstance(dice_list, list):
        dice_list = [dice_list]

    # If the name of the die is not a list, we turn it into a list of length 1
    if names is None:
        names = []
        for i in range(len(dice_list)):
            names.append('Die ' + str(i + 1))
    else:
        if not isinstance(names, list):
            names = [names]

    fig, ax = plt.subplots()

    # Create the list of sliders according to the list of parameters given
    sliders_list = []
    slider_height = 0.05
    slider_spacing = 0.00
    total_slider_size = len(dynamic_vars) * (slider_height + slider_spacing) + 0.1
    plt.subplots_adjust(bottom=total_slider_size)
    for i in range(len(dynamic_vars)):
        dynamic_var = dynamic_vars[i]
        slider_loc = slider_spacing * i + slider_height * i
        ax_sld = plt.axes([0.1, slider_loc, 0.8, slider_height])
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
        data = lambda mode: dice_utilities.get_data_for_plot(dice_list, mode[0], names)
    # There are dynamic variables - treat the dice as functions
    else:
        data = lambda mode: dice_utilities.get_data_for_plot(
            [die(*dice_utilities.extract_values_from_sliders(sliders_list)) for die in dice_list], mode[0], names)

    # The current mode that is shown. It is a list so it can be mutable
    chosen_mode = ['normal']
    # Attach the function of updating the plot when the slider changes
    for slider in sliders_list:
        slider.on_changed(lambda event:
                          dice_utilities.update_plot(fig, ax, lines_list, data(chosen_mode)))
        # slider.on_changed(lambda event: print(chosen_mode))

    # Create the list of line graphs for the different dice
    lines_list = []
    for i in range(len(dice_list)):
        # The plot is empty at first, but it will be populated later
        line, = ax.plot([], [], 'o-', linewidth=3, label='')
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
        dice_utilities.update_plot(fig, ax, lines_list, data(chosen_mode))

    normal_button.on_clicked(normal_button_click)

    # Button - At Least
    atleast_ax = plt.axes(
        [button_margin_x + (button_sz_x + button_spacing_x), 1 - button_margin_y, button_sz_x, button_sz_y])
    atleast_button = Button(atleast_ax, 'At Least', hovercolor=hovercolor)

    def atleast_button_click(event):
        chosen_mode[0] = 'atleast'
        dice_utilities.update_plot(fig, ax, lines_list, data(chosen_mode))

    atleast_button.on_clicked(atleast_button_click)

    # Button - At Most
    atmost_ax = plt.axes(
        [button_margin_x + (button_sz_x + button_spacing_x) * 2, 1 - button_margin_y, button_sz_x, button_sz_y])
    atmost_button = Button(atmost_ax, 'At Most', hovercolor=hovercolor)

    def atmost_button_click(event):
        chosen_mode[0] = 'atmost'
        dice_utilities.update_plot(fig, ax, lines_list, data(chosen_mode))

    atmost_button.on_clicked(atmost_button_click)

    # Simulate clicking on the 'normal' button to create the first plot
    normal_button_click(None)

    # General plot details
    ax.set_xlabel('Values')
    ax.set_ylabel('Probability')
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def print_summary(dice_list, x_list):
    """
    Plots basic information on a collection of dice
    :param dice_list: A list of dice
    :param x_list: Describes the values of the x-axis of the plot.
    """

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

    for j in range(4):
        header_text = ['Output', '#', string_list[j]]
        text = [header_text]
        for i in range(len(lists_list[j])):
            num_pipes = round(max_pipes * lists_list[j][i] / max_val)
            progress_bar_string = filled_char * num_pipes + empty_char * (max_pipes - num_pipes) + end_char
            text += [[x_list[i], '{:.2f}'.format(lists_list[j][i]), progress_bar_string]]
        print(tabulate(text, headers='firstrow') + '\n')


def plott(dice_list, x_list, param_list=None, draw_error_bars=False, title=None):
    """
    Plots basic information on a collection of dice
    :param dice_list: A 2D list of dice.
    dice_list[i][j] describes a die with a parameter value of param[i] and x value of x[j]
    :param x_list: Describes the values of the x-axis of the plot.
    Can be a list, or a 2D list, such that x_list[i] is the x-axis of the parameter param[i]
    :param param_list: Describes the values of parameters of the dice
    :param draw_error_bars: Describes if to draw error bars describing the deviation of the dice
    :param title: Optional title for the plot
    """

    if not isinstance(x_list[0], list):
        x_list = [x_list] * (1 if param_list is None else len(param_list))
    if (param_list is not None) and (not isinstance(param_list,list)):
        param_list = [param_list]
    if not isinstance(dice_list[0], list):
        dice_list = [dice_list]

    mean_list = dice_utilities.evaluate_dice_list(lambda d: d.mean(), dice_list)
    std_list = dice_utilities.evaluate_dice_list(lambda d: d.std(), dice_list)
    fig, ax = plt.subplots()
    if param_list is None:
        x = np.array(x_list[0])
        y = np.array(mean_list[0])
        err = np.array(std_list[0])
        if draw_error_bars:
            plt.errorbar(x, y, yerr=err, fmt='o-', linewidth=3, elinewidth=1, capsize=4)
        else:
            plt.plot(x, y, 'o-', linewidth=3)
    else:
        for i in range(len(param_list)):
            x = np.array(x_list[i])
            y = np.array(mean_list[i])
            err = np.array(std_list[i])
            if draw_error_bars:
                plt.errorbar(x, y, yerr=err, fmt='o-', linewidth=3, elinewidth=1, capsize=4, label=param_list[i])
            else:
                plt.plot(x, y, 'o-', linewidth=3, label=param_list[i])

    ax.set_xlabel('Values')
    ax.set_ylabel('Mean')
    ax.legend()
    ax.grid(alpha=0.3)
    if title is not None:
        ax.set_title(title)
    plt.show()
