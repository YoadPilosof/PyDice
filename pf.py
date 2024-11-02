import dice
import math

import pf


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


def basic_attack(hit_bonus, ac, success_dmg, crit_dmg=None, fail_dmg=None, crit_fail_dmg=None):
    roll_result = pf.check(dice.d(20), hit_bonus, ac)
    if crit_dmg is None:
        crit_dmg = success_dmg * 2
    if fail_dmg is None:
        fail_dmg = dice.zero()
    if crit_fail_dmg is None:
        crit_fail_dmg = dice.zero()

    return roll_result.switch(crit_fail_dmg, fail_dmg, success_dmg, crit_dmg)


def basic_save(save_bonus, dc, fail_dmg, crit_fail_dmg=None, success_dmg=None, crit_success_dmg=None):
    roll_result = pf.check(dice.d(20), save_bonus, dc)
    if crit_fail_dmg is None:
        crit_fail_dmg = fail_dmg * 2
    if success_dmg is None:
        success_dmg = (fail_dmg / 2).floor()
    if crit_success_dmg is None:
        crit_success_dmg = dice.zero()

    return roll_result.switch(crit_fail_dmg, fail_dmg, success_dmg, crit_success_dmg)