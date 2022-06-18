import dice
import math


def hit(roll, hit_bonus, ac, crit_thresold=20):
    """
    Checks if an attack is a hit or a miss
    :param roll: The d20 roll (including advantage or disadvanage, but not including modifiers)
    :param hit_bonus: The total bonus to hit
    :param ac: Target AC
    :param crit_thresold: Threshold for a critical hit, default is 20
    :return: A die with the values 0 (miss), 1 (hit) or 2 (crit)
    """

    def n_hit(n_roll, n_hit_bonus, n_ac, n_crit_threshold):
        if n_roll == 1:
            return 0
        if n_roll >= n_crit_threshold:
            return 2
        if n_roll + n_hit_bonus >= n_ac:
            return 1
        return 0

    return dice.func(n_hit, roll, hit_bonus, ac, crit_thresold)


def save(damage, saving_throw, DC, evasion=False):
    """
    Calculates the damage of a 'save for half damage' effect
    :param damage: Total damage
    :param saving_throw: The roll of the saving throw (with modifiers)
    :param DC: Save DC
    :param evasion: Boolean variable, if set to true, you take no damage on a success, and half on a fail. Default is False
    :return: Die describing the damage taken
    """

    if not evasion:
        return ((saving_throw >= DC).cond(0.5, 1) * damage).floor()
    else:
        return ((saving_throw >= DC).cond(0, 0.5) * damage).floor()


def get_pb(level):
    """
    Calculated a character's proficiency bonus based on their level
    :param level: The character's level
    :return: The character's proficiency bonus
    """
    return math.ceil(level / 4) + 1


def get_tier(level):
    """
    Calculates a character's tier based on their level
    :param level: The character's level
    :return: The character's tier
    """
    return 1 + (level >= 5) + (level >= 11) + (level >= 17)
