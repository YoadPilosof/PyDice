import dice
import math

@dice.func
def check(roll, bonus, dc):
    """
    Checks if a check is a success of a failure
    :param roll: The d20 roll (including advantage or disadvanage, but not including modifiers)
    :param bonus: The total bonus to the check
    :param dc: Check dc
    :return: A die with the values 0 (critical fail), 1 (fail), 2 (success) or 3 (critical success)
    """

    total_roll = roll + bonus
    if total_roll < dc - 10:
        result = 0
    elif total_roll < dc:
        result = 1
    elif total_roll < dc + 10:
        result = 2
    else:
        result = 3

    if roll == 1:
        result = max(result - 1, 0)
    elif roll == 20:
        result = min(result + 1, 3)

    return result
