import dice
import dice_utilities
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np


def general_examples():
    def attack(roll, defense, hitbonus):
        if roll == 1:
            return 0
        if roll == 20:
            return 2
        if roll + hitbonus >= defense:
            return 1
        return 0

    def damage(result, miss_dmg, hit_dmg, crit_dmg):
        if result == 0:
            return miss_dmg
        if result == 1:
            return hit_dmg
        if result == 2:
            return crit_dmg
        return 0

    def diminishing(roll, size):
        if roll == size and size > 4:
            return dice.func(diminishing, dice.d(size - 2), size - 2) + roll
        return roll

    def blades(*args):
        if max(*args) == 6:
            return 2 + (args.count(6) > 1)
        return 0 + (max(*args) >= 4)

    d4 = dice.d(4)
    d6 = dice.d(6)
    d8 = dice.d(8)
    d10 = dice.d(10)
    d12 = dice.d(12)
    d20 = dice.d(20)

    # d = d8 ; name = '1d8'
    # d = d4 + d6 + d8 ; name = '1d4 + 1d6 + 1d8'
    # d = d6 ** 8 ; name = '8d6'
    d = d20.adv() ; name = '1d20 with advantage'
    # d = d20.dis(3) ; name = '1d20 with triple disadvantage'
    # d = dice.drop_pos([d6]*4, -1) ; name = '4d6 drop lowest'
    # d = d6.exp() ; name = 'exploding 1d6'
    # d = ~((d6 >= 5) & (d4 == 4)) ; name = '~((d6 >= 5) & (d4 == 4))'
    # d = d6.count([3, 4, 5, 6], 6) ; name = 'Count 3-6, with 6 being double'
    # d = d6.count(3,4,5,6,6) ; name = 'Count 3-6, with 6 being double'
    # d = dice.binary(0.65).cond(d8 + 3, 0) ; name = '65% chance to hit'
    # d = d6.reroll_on(1, max_depth=1) ; name = '1d6 reroll 1s'
    # d = d6.reroll_on(1, max_depth=2) ; name = '1d6 reroll 1s (depth = 2)'
    # d = d6.reroll_on(1, max_depth='inf') ; name = '1d6 reroll 1s (depth = infinity)'
    # hit = dice.func(attack, d20.adv(), 20, 10)
    # dmg = dice.func(damage, hit, 0, d10 + 5, d10 ** 2 + 5);
    # d = dmg;
    # name = 'Calculate damage'
    # d = dice.func(diminishing, d10, 10) ; name = 'Diminishing exploding die'
    # d = dice.func(blades, *([d6] * 4)) ; name = 'Blades die'
    # d = dice.list2dice([0, 1, 2, 2]).switch(d4, d6, d8) ; name = 'Switch case'

    # d.print_normal(name)
    # d.print_at_least(name)
    # d.print_at_most(name)

    dice1 = d12
    dice2 = d6 ** 2
    dice3 = d4 ** 3
    # dice.plot([dice1, dice2, dice3], ['d12', '2d6', '3d4'], title='Splitting Dice')

    d_1 = lambda AC: (d20.adv() + 5 >= AC).cond(d8 + 3, 0)
    d_2 = lambda AC: (d20 + 5 >= AC).cond(d8 + 3, 0)
    d_3 = lambda AC: (d20.dis() + 5 >= AC).cond(d8 + 3, 0)
    dice.plot([d_1, d_2, d_3], names=['Advantage', 'Normal', 'Disadvantage'], dynamic_vars=[('AC', 10, 15, 20, 1)])
    # dice.plot([d_1, d_2, d_3], dynamic_vars=[('AC', 10, 15, 20, 1)])

    d_1 = lambda Hit, AC: (d20.adv() + Hit >= AC).cond(d8 + 3, 0)
    d_2 = lambda Hit, AC: (d20 + Hit >= AC).cond(d8 + 3, 0)
    d_3 = lambda Hit, AC: (d20.dis() + Hit >= AC).cond(d8 + 3, 0)
    # dice.plot([d_1, d_2, d_3], names=['Advantage', 'Normal', 'Disadvantage'], dynamic_vars=[
    #     ('Hit', 0, 5, 10, 1), ('AC', 10, 15, 20, 1)])

    high_d4 = [dice.get_pos([d4] * (i + 1), 0) for i in range(5)]
    high_d6 = [dice.get_pos([d6] * (i + 1), 0) for i in range(5)]
    high_d8 = [dice.get_pos([d8] * (i + 1), 0) for i in range(5)]

    dice_list = [high_d4, high_d6, high_d8]
    x_list = [1, 2, 3, 4, 5]
    # dice.print_summary(high_d4, x_list)
    param_list = ['d4', 'd6', 'd8']
    # dice.plott(high_d4, x_list, 'd4', draw_error_bars=True, title='Keeping the highest cube')
    # dice.plott(dice_list, x_list, param_list, draw_error_bars=True,title='Keeping the highest cube')

def Theyandor():
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()

    def attack(roll, hitbonus, ac):
        if roll == 20:
            return 2
        if roll == 1:
            return 0
        if roll + hitbonus >= ac:
            return 1
        return 0

    def two_attack_dmg(hit1, hit2):
        total_dmg = dice.zero()
        used_d4 = 1
        match hit1:
            case 0:
                total_dmg += 0
            case 1:
                total_dmg += d10 + d4 + dex
                used_d4 = 0
            case 2:
                total_dmg += d10**2 + d4**2 + dex
                used_d4 = 0
        match hit2:
            case 0:
                total_dmg += 0
            case 1:
                total_dmg += d10 + d4*used_d4 + dex
            case 2:
                total_dmg += d10**2 + d4**2*used_d4 + dex
        return total_dmg

    dex = +4
    prof = +3
    two_attack_dmg_list = []
    for ac in range(10, 21):
        hit = dice.func(attack, d20.adv(), dex + prof, ac)
        two_hit_dmg = dice.func(two_attack_dmg, hit, hit)
        two_attack_dmg_list.append(two_hit_dmg)
    dice.plott(two_attack_dmg_list, list(range(10, 21)))


if __name__ == '__main__':
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    general_examples()
    # Theyandor()


