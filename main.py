import dice
import time
import dnd
import math


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

    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()

    time_start = time.time()
    # d = d8 ; name = '1d8'
    # d = d4 + d6 + d8 ; name = '1d4 + 1d6 + 1d8'
    # d = d6 ** 8 ; name = '8d6'
    # d = d20.adv() ; name = '1d20 with advantage'
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
    # d = dice.get_pos([dice.drop_pos([d6]*4, -1)]*6, 1)
    # def high_low(x, y):
    #     if x > y:
    #         return x, y
    #     return y, x
    # d_list = dice.func(high_low, d6, d6) ; name = 'A tuple returning function'
    # d = d_list[0]
    # d = d6.get_pos(11, 20) ; name = "Optimized advantage"
    # d = dice.d(6, 100) ; name = "Optimized Multidice"
    # d = d6**100

    time_end = time.time()

    # d.print_normal(name)
    # d.print_at_least(name)
    # d.print_at_most(name)
    # print(d.std())
    print("Elapsed Time = {} us".format(1000 * 1000 * (time_end - time_start)))

    dice1 = d12
    dice2 = d6 ** 2
    dice3 = d4 ** 3
    # dice.plot([dice1, dice2, dice3], names=['d12', '2d6', '3d4'], title='Splitting Dice')

    d_1 = lambda AC: (d20.adv() + 5 >= AC).cond(d8 + 3, 0)
    d_2 = lambda AC: (d20 + 5 >= AC).cond(d8 + 3, 0)
    d_3 = lambda AC: (d20.dis() + 5 >= AC).cond(d8 + 3, 0)
    # dice.plot([d_1, d_2, d_3], names=['Advantage', 'Normal', 'Disadvantage'], dynamic_vars=[('AC', 10, 15, 20, 1)])
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
    # dice.plott(high_d4, x_list, 'd4', draw_error_bars=True, xlabel='Number of Dice', title='Keeping the highest cube')
    # dice.plott(dice_list, x_list, param_list, draw_error_bars=True, xlabel='Number of Dice', title='Keeping the highest cube')

    Xd4 = lambda x: d4 ** x
    # dice.plot(Xd4, names='Xd4', dynamic_vars=[('X', 1, 1, 20, 1)])

    one_stat = dice.drop_pos([d6] * 4, -1)
    # one_stat.adv(6).print_normal()
    stats = []
    # for i in range(6):
    #     stats.append(one_stat.get_pos(i, 6))
    # stats[1].print_normal()

    # dice.highest(d4,d6,d8,d10,d12).print_normal()

    # d8.get_pos(list(range(10)), 20).print_normal()


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
                total_dmg += d10 ** 2 + d4 ** 2 + dex
                used_d4 = 0
        match hit2:
            case 0:
                total_dmg += 0
            case 1:
                total_dmg += d10 + d4 * used_d4 + dex
            case 2:
                total_dmg += d10 ** 2 + d4 ** 2 * used_d4 + dex
        return total_dmg

    dex = +4
    prof = +3
    two_attack_dmg_list = []
    for ac in range(10, 21):
        hit = dice.func(attack, d20.adv(), dex + prof, ac)
        two_hit_dmg = dice.func(two_attack_dmg, hit, hit)
        two_attack_dmg_list.append(two_hit_dmg)
    dice.plott(two_attack_dmg_list, list(range(10, 21)))


def hold_person():
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    SaveDC = 15
    WisSaveBonus = +3
    MindSliver = d4
    UnsettlingWords = d6

    default_basic = d20 + WisSaveBonus >= SaveDC
    default_recursive = default_basic.count_attempts(10) - 1
    stack = (d20 + WisSaveBonus - MindSliver - UnsettlingWords >= SaveDC) \
        .cond(0, default_recursive + 1)
    split = (d20 + WisSaveBonus - MindSliver >= SaveDC) \
        .cond(0, (d20 + WisSaveBonus - UnsettlingWords >= SaveDC)
              .cond(0, default_recursive + 1) + 1)

    stack.print_normal("Stacked")
    split.print_normal("Split")
    # default_basic.print_normal()


def attack_options():
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    light_attack = lambda ac, str, prof, extra: \
        dnd.hit(d20, str + prof * 2, ac).switch(0, d8 + extra, d8 ** 2 + extra) << "Light Attack"
    normal_attack = lambda ac, str, prof, extra: \
        dnd.hit(d20, str + prof, ac).switch(0, d8 + str + extra, d8 ** 2 + str + extra) << "Normal Attack"
    heavy_attack = lambda ac, str, prof, extra: \
        dnd.hit(d20, str, ac).switch(0, d8 + str + prof * 2 + extra, d8 ** 2 + str + prof * 2 + extra) << "Heavy Attack"
    str_slider = ['STR Mod', 2, 4, 6, 1]
    prof_slider = ['Proficiency Bonus', 2, 3, 6, 1]
    ac_slider = ['Target AC', 10, 15, 30, 1]
    extra_slider = ['Extra Damage', 0, 0, 20, 1]
    dice.plot_mean([light_attack, normal_attack, heavy_attack],
                   [ac_slider, str_slider, prof_slider, extra_slider],
                   title='Attack Options')


def attack_options_rogue():
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    light_attack = lambda ac, str, prof, sneak: \
        dnd.hit(d20.adv(), str + prof * 2, ac).switch(0, d8 + str + d4 * sneak,
                                                      d8 * 2 + str + d4 * sneak * 2) << "Light Attack"
    normal_attack = lambda ac, str, prof, sneak: \
        dnd.hit(d20.adv(), str + prof, ac).switch(0, d8 + str + d6 * sneak,
                                                  d8 * 2 + str + d6 * sneak * 2) << "Normal Attack"
    heavy_attack = lambda ac, str, prof, sneak: \
        dnd.hit(d20.adv(), str, ac).switch(0, d8 + str + d8 * sneak, d8 * 2 + str + d8 * sneak * 2) << "Heavy Attack"
    str_slider = ['STR Mod', 2, 4, 6]
    prof_slider = ['Proficiency Bonus', 2, 3, 6]
    ac_slider = ['Target AC', 10, 15, 30]
    sneak_attack_slider = ['Sneak Attack Dice', 1, 1, 10]
    dice.plot_mean([light_attack, normal_attack, heavy_attack],
                   [ac_slider, str_slider, prof_slider, sneak_attack_slider],
                   title='Attack Options')


def class_comparison():
    def warlock(level, ac, turns):
        prof = dnd.get_pb(level)
        mod = 3 + (level >= 4) + (level >= 8)
        num_attacks = 1 + (level >= 5) + (level >= 11) + (level >= 17)
        agonizing = mod if level >= 2 else 0
        damage_per_attack = dnd.hit(d20, mod + prof, ac).switch(0,
                                                                d10 + d6 + agonizing,
                                                                d10 ** 2 + d6 ** 2 + agonizing)
        DPR = damage_per_attack * num_attacks
        return DPR << "Warlock (Hex + Agonizing Blast)"

    def fighter(level, ac, turns):
        prof = dnd.get_pb(level)
        mod = 3 + (level >= 4) + (level >= 6)
        num_attacks = 1 + (level >= 5) + (level >= 11) + (level >= 20)
        crit_threshold = 20 - (level >= 3) - (level >= 15)
        action_surges = (level >= 2) + (level >= 17)
        damage_per_attack = dnd.hit(d20, mod + prof, ac, crit_threshold).switch(0,
                                                                                (d8 + 2 + mod),
                                                                                (d8 ** 2 + 2 + mod))
        DPR = damage_per_attack ** num_attacks
        total_damage = DPR * (turns + min(turns, action_surges))
        return total_damage / turns << "Fighter (Champion, Longsword + Dueling)"

    def rogue(level, ac, turns):
        prof = dnd.get_pb(level)
        mod = 3 + (level >= 4) + (level >= 8)
        sneak_attack = d6 * math.ceil(level / 2)
        base_roll = d20.adv() if level >= 2 else d20
        DPR = dnd.hit(base_roll, mod + prof, ac).switch(0,
                                                        d6 + sneak_attack + mod,
                                                        d6 ** 2 + sneak_attack ** 2 + mod)
        return DPR << "Rogue (Advantage + Sneak Attack)"

    def monk(level, ac, turns):
        prof = dnd.get_pb(level)
        mod = 3 + (level >= 4) + (level >= 8)
        MartialArtsDieSize = 2 + 2 * dnd.get_tier(level)
        MartialArtsDie = dice.d(MartialArtsDieSize)
        WeaponDie = MartialArtsDie if level >= 5 else d6
        ki = level if level >= 2 else 0
        number_attacks = 2 if level >= 5 else 1
        damage_per_action_attack = dnd.hit(d20, mod + prof, ac).switch(0,
                                                                       WeaponDie + mod,
                                                                       WeaponDie ** 2 + mod)
        damage_per_bonus_attack = dnd.hit(d20, mod + prof, ac).switch(0,
                                                                      MartialArtsDie + mod,
                                                                      MartialArtsDie ** 2 + mod)
        total_damage = damage_per_action_attack * (number_attacks * turns) + \
                       damage_per_bonus_attack * turns + \
                       damage_per_bonus_attack * min(ki, turns)
        return total_damage / turns << "Monk (Only flurry of blows)"

    level_slider = ['Level', 1, 1, 20]
    ac_slider = ['AC', 10, 15, 25]
    turns_slider = ['Number of Turns', 1, 1, 10]

    dice.plot_mean([warlock, fighter, rogue, monk], [level_slider, ac_slider, turns_slider],
                   title="Damage Per Round Comparison",
                   y_lim='max')


def acquire_funds(pc = "ori", prof=2, type = "safe"):
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()

    # Ori
    if pc == "ori":
        cha = +0
        int = -1
        wis = -1
        insight = wis + 2 * prof
        persuasion = cha + 2 * prof
        history = int + 2 * prof
        deception = cha + 2 * prof
        intimidation = cha + 2 * prof

    # Yoad
    if pc == "yoad":
        cha = +5
        int = +1
        wis = +0
        insight = wis + prof
        persuasion = cha + 2 * prof
        history = int + 2 * prof
        deception = cha + 2 * prof
        intimidation = cha + math.floor(prof / 2)

    successes_play_it_safe = (d20 + insight > (d10 ** 2)) + \
                             (d20 + persuasion > (d10 ** 2)) + \
                             (d20 + history > (d10 ** 2))

    successes_take_a_risk  = (d20 + insight > (d10 ** 2 + 5)) + \
                             (d20 + deception > (d10 ** 2 + 5)) + \
                             (d20 + intimidation > (d10 ** 2 + 5))

    play_it_safe = successes_play_it_safe.switch(+0.5, +1, +1.25, +1.5) - 1 << "Play it Safe"
    take_a_risk = successes_take_a_risk.switch(-1, +0.5, +1.5, +2) - 1 << "Take a Risk"

    # play_it_safe.print_normal()
    # take_a_risk.print_normal()
    if type == "safe":
        return play_it_safe
    if type == "risk":
        return take_a_risk


if __name__ == '__main__':
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    # dice.print_summary([d4 << "1d4", d6 << "1d6", d8 << "1d8", d10 << "1d10", d12 << "1d12", d20 << "1d20", d100 << "1d100"], 0)
    d10s = lambda N: d10 ** N << "Nd10"
    d8s = lambda N: d8 ** N << "Nd8"
    d6s = lambda N: d6 ** N << "Nd6"
    sld = ['Number of Dice', 1, 1, 6]
    # dice.plot_stats([d6 ** 4 << "4d6", d8 ** 4 << "4d8", d10 ** 4 << "4d10"])
    # dice.plot_stats([d6s, d8s, d10s], sld)

    # general_examples()
    # Theyandor()
    # hold_person()
    # attack_options_rogue()
    class_comparison()
    # acquire_funds(2)
    rerolled = lambda x : d12.reroll_comp('<=', x) << "Reroll on " + str(x) + " or lower"
    # dice.plot_stats(rerolled, ["Reroll LT", 0, 1, 12], y_lim='max')
    # TODO: Add non-auto y-min to plot

