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
        for roll, chance in self.pdf.items():
            res += roll * chance
        return res

    def std(self):
        """
        :return: Returns the standard deviation of the dice
        """

        # Calculate the variance, and later take the square root
        var = 0
        # First find the mean, as it is used in the definition of the variance
        m = self.mean()
        for roll, chance in self.pdf.items():
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
        Simulates a roll of the die
        :param n: How many dice rolls to simulate
        :return: The simulates roll(s). If a value was given to n, returns a list, otherwise returns a value
        """
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

        # Force the <other> die to be a die object
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

    def __sub__(self, other):
        """
        Creates a die describing the difference of two dice
        :param other: The second die to subtract. Can be a number
        :return: A new die, with statistics according to the difference of the two dice
        """

        # Force the <other> die to be a die object
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

    def __mul__(self, other):
        """
        Creates a die describing the product of two dice
        :param other: The second die to multiply. Can be a number
        :return: A new die, with statistics according to the product of the two dice
        """

        # Force the <other> die to be a die object
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

    def __truediv__(self, other):
        """
        Creates a die describing the floating point division of two dice
        :param other: The second die to divide by. Can be a number
        :return: A new die, with statistics according to the division of the two dice
        """

        # Force the <other> die to be a die object
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

    def __mod__(self, other):
        """
        Creates a die describing the modulus of two dice
        :param other: The second die to perform modulus by. Can be a number
        :return: A new die, with statistics according to the modulus of the two dice
        """

        # Force the <other> die to be a die object
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

    def __floordiv__(self, other):
        """
        Creates a die describing the integer division of two dice
        :param other: The second die to divide by. Can be a number
        :return: A new die, with statistics according to the division of the two dice
        """

        # Force the <other> die to be a die object
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

    def __pow__(self, power, modulo=None):
        """
        Creates a die describing rolling the same die multiple times and taking the sum
        :param power: How many dice to roll
        :return: A new die, with statistics according to the sum of the die
        """
        if power == 1:
            return copy.deepcopy(self)
        else:
            # return (self ** (power - 1)) + copy.deepcopy(self)
            if power % 2:
                half_power_smaller = power // 2
                half_power_larger = power - (power // 2)
                half_die_smaller = self ** half_power_smaller
                half_die_larger = self ** half_power_larger
                return half_die_smaller + half_die_larger
            else:
                half_amount = self ** (power / 2)
                return half_amount + half_amount

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
        :return: A new die, describing rolling the current die n times, and taking the highest result
        """
        if n == 1:
            return self
        else:
            return self.get_pos(0, n)

    def dis(self, n=2):
        """
        :return: A new die, describing rolling the current die n times, and taking the lowest result
        """
        if n == 1:
            return self
        else:
            return self.get_pos(n-1, n)

    def exp(self, explode_on=None, max_depth=2):
        """
        Simulates exploding a die, i.e., rolling the die, and if it rolls the highest value, roll it again
        :param explode_on: Optional parameter. If given, the die will explode on this value, instead of the maximum value of the die
        :param max_depth: Maximum depth of the simulation. Default is 2
        :return: A new die
        """

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

    def count(self, *args):
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

    def cond(self, die_if_true, die_if_false):
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

    def switch(self, *args):
        """
        Uses the first die (which rolls integers between 0 and N-1) to choose a second die
        :param args: A list or tuple of the other set of dice
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

        new_dice = dice()

        for self_roll, self_chance in self.pdf.items():
            for other_roll, other_chance in other_dice_list[self_roll].pdf.items():
                # If this roll is not a possible roll of the new dice, add it with probability of 0
                if other_roll not in new_dice.pdf.keys():
                    new_dice.pdf[other_roll] = 0
                # Increase the chance of getting this outcome by the product of the probabilities of each die
                new_dice.pdf[other_roll] += self_chance * other_chance

        return new_dice

    def dictionary(self, roll_dictionary, default_value=None):
        """
        Transforms the die according to a dictionary
        :param roll_dictionary: A dictionary describing the transformation, has <float> input and <die>/<float> output
        :param default_value: Optional parameter.
        If given, values that do not have a dictionary value will get this value instead
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
                new_value = dice_utilities.force_cube(new_value)
                for new_roll, new_chance in new_value.pdf.items():
                    if not (new_roll in new_dice.pdf):
                        new_dice.pdf[new_roll] = 0
                    new_dice.pdf[new_roll] += chance * new_chance
            else:
                if default_value is None:
                    # If the dictionary doesn't have information about this roll, and there is no default value,
                    # raise an error
                    raise ()
                else:
                    new_value = default_value
                    # Transform the new roll into a die if it is not already
                    new_value = dice_utilities.force_cube(new_value)
                    for new_roll, new_chance in new_value.pdf.items():
                        if not (new_roll in new_dice.pdf):
                            new_dice.pdf[new_roll] = 0

        return new_dice

    def get_pos(self, pos_index, num_dice):
        """
        Simulates rolling the <self> die <num_dice> times, and taking the <pos_index>th largest value
        :param pos_index: Which die to take. Can be a list
        :param num_dice: How many dice were rolled
        :return: The new die
        """

        # Transform the pos_index so negative values are counted from the end of the list
        if isinstance(pos_index, list):
            pos_index = [index % num_dice for index in pos_index]
        else:
            pos_index = pos_index % num_dice

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

    def drop_pos(self, pos_index, num_dice):
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

    def round(self):
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
            new_dice.pdf[new_roll] += chance * chance

        # Return the new die
        return new_dice

    def floor(self):
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
            new_dice.pdf[new_roll] += chance * chance

        # Return the new die
        return new_dice

    def ceil(self):
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
            new_dice.pdf[new_roll] += chance * chance

        # Return the new die
        return new_dice

    # ~~~~~~~~~ Overloaded Comparative Operations ~~~~~~~~~

    def __lt__(self, other):
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

    def __le__(self, other):
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

    def __eq__(self, other):
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

    def __ne__(self, other):
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

    def __gt__(self, other):
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

    def __ge__(self, other):
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

    def __and__(self, other):
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

    def __or__(self, other):
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

    def __xor__(self, other):
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

    def __invert__(self):
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

    def reroll_func(self, lambda_cond, max_depth=1):
        """
        A general reroll function. Simulates rolling a die, and rerolling the dice if a certain condition is met
        :param lambda_cond: A lambda function that takes a number and returns a boolean.
        :param max_depth: A parameter describing if the dice is rerolled multiple times if a bad result is continuously rolled
        Can be an integer (1 for only one reroll), or 'inf' for an infinite number of rerolls
        :return: A new die
        """

        new_dice = dice()

        # If max_depth is 'inf', we calculate the statistics of the new die mathematically,
        # by using conditional probability (i.e. pdf of <self> given lambda_func(<self>) = False)
        if max_depth == 'inf':
            self_rolls = list(self.pdf.keys())
            self_chances = list(self.pdf.values())
            # The total probability of rolling a value that is not rerolled
            total_prob = sum([self_chances[i] for i in range(len(self_rolls)) if not lambda_cond(self_rolls[i])])

            for i in range(len(self_rolls)):
                # We only add the value to the new die if it is not rerolled
                if not lambda_cond(self_rolls[i]):
                    # The chance to roll this new value, is the original chance,
                    # normalized so the chance of rolling something is 1
                    new_dice.pdf[self_rolls[i]] = self_chances[i] / total_prob

        # If max_depth is no 'inf', we calculate the statistics of the new die recursively
        else:
            # Recursion stop case, if max_depth is 0, we don't reroll, regardless of the roll
            if max_depth == 0:
                return self

            # Recursively call the reroll function with a smaller max_depth
            deeper_dice = self.reroll_func(lambda_cond, max_depth=max_depth - 1)

            for self_roll, self_chance in self.pdf.items():
                # If the value roll for the <self> die is a value we reroll on, consider the <deeper_dice> die
                if lambda_cond(self_roll):
                    for deeper_dice_roll, deeper_dice_chance in deeper_dice.pdf.items():
                        # The outcome of these two rolls, is the <deeper_dice> roll
                        # If this roll is not a possible roll of the new dice, add it with probability of 0
                        if not (deeper_dice_roll in new_dice.pdf):
                            new_dice.pdf[deeper_dice_roll] = 0
                        # Increase the chance of getting this outcome by the product of the probabilities of each die
                        new_dice.pdf[deeper_dice_roll] += self_chance * deeper_dice_chance

                # If the <self> die was not rerolled, ignore the <deeper_dice> die
                else:
                    # The outcome of these two rolls, is just first value rolled (since it was not rerolled)
                    new_roll = self_roll
                    # If this roll is not a possible roll of the new dice, add it with probability of 0
                    if not (new_roll in new_dice.pdf):
                        new_dice.pdf[new_roll] = 0
                    # Increase the chance of getting this outcome by the product of the probability of the first die
                    new_dice.pdf[new_roll] += self_chance

        return new_dice

    def reroll_on(self, match_list, max_depth=1):
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


# ~~~~~~~~~ Custom Dice Constructors ~~~~~~~~~

def d(size, n=1):
    """
    Constructs a new die, which is uniformly distributed over 1 .. size
    :param size: The size of the die
    :param n: How many dice are rolled
    :return: The new die
    """
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


def standard_dice():
    """
    A helpful function that generates the 7 basic RPG dice. Use:
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


def from_const(val):
    """
    Constructs a trivial die with only one value
    :param val: The value of the die
    :return: The new die
    """
    new_dice = dice()
    new_dice.pdf[val] = 1
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
        # If elem isn't in the new die dictionary, set its probability to 0
        if not (elem in new_dice.pdf):
            new_dice.pdf[elem] = 0
        # Since all rolls are equiprobable, the chance to get any result is 1 / total number of options
        new_dice.pdf[elem] += 1 / len(values_list)

    return new_dice

    # ~~~~~~~~~ Dice Functions ~~~~~~~~~


def range2dice(*args):
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


def get_pos(dice_list, index_list=None):
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

    # We first perform modulus on each index, so -1 will turn to len(list)-1
    mod_index_list = [(index_list[i] % len(dice_list)) for i in range(len(index_list))]
    # We convert the list of indices to drop, to a list of indices to keep
    keep_index_list = [i for i in range(len(dice_list)) if not (i in mod_index_list)]
    # We call the get_pos method with the list of indices to keep
    return get_pos(dice_list, keep_index_list)


def func(lambda_func, *args):
    """
    Executes a generic function on dice
    :param lambda_func: The function to execute
    :param args: The dice and values to pass to the function
    :return: A new die, according to the function
    """

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
            value = lambda_func(*comb[1])
            value = dice_utilities.force_cube(value)

            comb_chance = comb[0]

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
        # If we are given only one argument, and it is a list, treat it as though we received a tuple
        if isinstance(args[0], list):
            args = tuple(args[0])
        # If we are only given one argument, and it is a die or a number, we have reached the recursion stop case
        # The highest value of one argument, is the argument, so we simply return it
        else:
            return args[0]

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


def lowest(*args):
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
            return args[0]

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


# ~~~~~~~~~ Plotting ~~~~~~~~~


def plot(dice_list, dynamic_vars=None, names=None, xlabel='Value', title=None):
    """
    Plots statistics of dice
    :param dice_list: A die object, or a list of dice objects to plot the statistics of
    :param dynamic_vars: A list of tuples, describing dynamic variables of the plot.
    If this list is not empty, dice_list should instead be a list of lambda functions which generate the dice
    The tuple is in the form: <name>, <min_val>, <start_val>, <max_val>, <val_step>
    :param names: A string or list of strings. If given, override the current names of the dice
    :param xlabel: Label for x-axis
    :param title: Optional title for the plot, replaces the default one
    """

    if dynamic_vars is None:
        dynamic_vars = []

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
    ax.set_xlabel(xlabel)
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


def plott(dice_list, x_list, param_list=None, draw_error_bars=False, xlabel='Value', title=None):
    """
    Plots basic information on a collection of dice
    :param dice_list: A 2D list of dice.
    dice_list[i][j] describes a die with a parameter value of param[i] and x value of x[j]
    :param x_list: Describes the values of the x-axis of the plot.
    Can be a list, or a 2D list, such that x_list[i] is the x-axis of the parameter param[i]
    :param param_list: Describes the values of parameters of the dice
    :param draw_error_bars: Describes if to draw error bars describing the deviation of the dice
    :param xlabel: Label of x-axis
    :param title: Optional title for the plot
    """

    if not isinstance(x_list[0], list):
        x_list = [x_list] * (1 if param_list is None else len(param_list))
    if (param_list is not None) and (not isinstance(param_list, list)):
        param_list = [param_list]
    if not isinstance(dice_list[0], list):
        dice_list = [dice_list]

    mean_list = dice_utilities.evaluate_dice_list(lambda die: die.mean(), dice_list)
    std_list = dice_utilities.evaluate_dice_list(lambda die: die.std(), dice_list)
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean')
    ax.legend()
    ax.grid(alpha=0.3)
    if title is not None:
        ax.set_title(title)
    plt.show()
