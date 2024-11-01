import random

import pf
import tabulate

import dice
import time
import dnd
import math
import numpy as np


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


def pf2e_map():
    def two_attacks(ac, bonus, agile, num_of_actions):
        bonus_by_moving = +2
        first = pf.check(d20, bonus + bonus_by_moving, ac).switch(0, 0, 1, 2)
        if num_of_actions == 2:
            return first << "Moving to flank"

        if agile:
            second = pf.check(d20, bonus + bonus_by_moving - 4, ac).switch(0, 0, 1, 2)
        else:
            second = pf.check(d20, bonus + bonus_by_moving - 5, ac).switch(0, 0, 1, 2)

        return first + second << "Moving to flank"

    def three_attacks(ac, bonus, agile, num_of_actions):
        first = pf.check(d20, bonus, ac).switch(0, 0, 1, 2)
        if agile:
            second = pf.check(d20, bonus - 4, ac).switch(0, 0, 1, 2)
            third = pf.check(d20, bonus - 8, ac).switch(0, 0, 1, 2)
        else:
            second = pf.check(d20, bonus - 5, ac).switch(0, 0, 1, 2)
            third = pf.check(d20, bonus - 10, ac).switch(0, 0, 1, 2)

        if num_of_actions == 2:
            return first + second << "Only attacking"
        return first + second + third << "Only attacking"

    ac_slider = ["AC", 10, 10, 25, 1]
    bonus_slider = ["Hit Bonus", 0, 0, 15, 1]
    num_of_actions_slider = ["# Actions", 2, 3, 3, 1]
    is_agile_slider = ["Is agile?", 0, 0, 1, 1]

    dice.plot_mean([two_attacks, three_attacks], [ac_slider, bonus_slider, is_agile_slider, num_of_actions_slider],
                   title="Attack options comparison")

def pf2e_treat_wounds():
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()

    Level = [i+1 for i in range(20)]
    Wis = [(4 if i < 10 else 5) for i in Level]
    Prof = [2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8]

    TotalBonus = [l+w+p for l,w,p in zip(Level,Wis,Prof)]

    def TreatWounds(lvl, min_lvl, dc, bonus_heal, assurance, hour):
        if lvl < min_lvl:
            return dice.zero()
        roll = dice.from_const(10) if assurance else d20
        check_result = pf.check(roll, TotalBonus[lvl-1], dc)
        healing = check_result.switch(-d8, 0, (d8**2 + bonus_heal) * (1 + hour), (d8**4 + bonus_heal) * (1 + hour))
        return healing

    Trained = lambda lvl, assurance, hour: TreatWounds(lvl, 1, 15, 0, assurance, hour) << "Trained"
    Expert = lambda lvl, assurance, hour: TreatWounds(lvl, 3, 20, 10, assurance, hour) << "Expert"
    Master = lambda lvl, assurance, hour: TreatWounds(lvl, 7, 30, 30, assurance, hour) << "Master"
    Legendary = lambda lvl, assurance, hour: TreatWounds(lvl, 15, 40, 50, assurance, hour) << "Legendary"

    dice.plot_mean([Trained, Expert, Master, Legendary], [["Level", 1, 1, 20], ["Assurance", 0, 0, 1], ["Hour Long", 0, 0, 1]], y_lim='max')


def pf2e_scout(bonus):
    d20 = dice.d(20)
    d6 = dice.d(6)

    check_result = pf.check(d20, bonus, 15)
    # return check_result.switch(0, 1, 0, 0)
    return check_result.switch(d6 <= 5, d6 <= 2, d6 <= 1, 0) << "Chance to Veer"

def pf2e_hunted_shot_vs_hunters_aim():
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()

    damage_die = d6
    precision_die = d8
    support_benefit = d8
    # support_benefit = dice.zero()

    def hunted_shot(hit_bonus, target_ac):
        hit_chance1 = pf.check(d20, hit_bonus, target_ac)
        hit_chance2 = pf.check(d20, hit_bonus - 5, target_ac)

        def calc_damage_for_two_hits(hit1, hit2):
            dmg_so_far = dice.zero()
            if hit1 == 2:
                dmg_so_far += damage_die + precision_die + support_benefit
            if hit1 == 3:
                dmg_so_far += damage_die*2 + precision_die*2 + support_benefit

            if hit2 == 2:
                dmg_so_far += damage_die + (precision_die * (hit1 <= 1)) + support_benefit
            if hit2 == 3:
                dmg_so_far += damage_die*2 + (precision_die*2 * (hit1 <= 1)) + support_benefit

            return dmg_so_far

        return dice.func(calc_damage_for_two_hits, hit_chance1, hit_chance2) << "Hunted Shot"

    def hunters_aim(hit_bonus, target_ac):
        return pf.check(d20, hit_bonus + 20, target_ac).switch(0, 0, damage_die + precision_die + support_benefit, damage_die*2 + precision_die*2 + support_benefit) << "Hunters Aim"

    dice.plot_mean([hunted_shot, hunters_aim], dynamic_vars=[["Hit Bonus", 0, 0, 20], ["Target AC", 10, 10, 30]])

if __name__ == '__main__':
    d4, d6, d8, d10, d12, d20, d100 = dice.standard_dice()
    # dice.print_summary([d4 << "1d4", d6 << "1d6", d8 << "1d8", d10 << "1d10", d12 << "1d12", d20 << "1d20", d100 << "1d100"], 0)
    # d10s = lambda N: d10 ** N << "Nd10"
    # d8s = lambda N: d8 ** N << "Nd8"
    # d6s = lambda N: d6 ** N << "Nd6"
    # sld = ['Number of Dice', 1, 1, 6]
    # dice.plot_stats([d6 ** 4 << "4d6", d8 ** 4 << "4d8", d10 ** 4 << "4d10"])
    # dice.plot_stats([d6s, d8s, d10s], sld)

    # general_examples()
    # Theyandor()
    # hold_person()
    # attack_options_rogue()
    # class_comparison()
    # acquire_funds(2)
    # rerolled = lambda x : d12.reroll_comp('<=', x) << "Reroll on " + str(x) + " or lower"
    # dice.plot_stats(rerolled, ["Reroll LT", 0, 1, 12], y_lim='max')
    # upgrade_weapons_v2_overview()
    # compare_flash_strike()
    # dice.plot_mean(godshot_hp, [["Hit Die Size", 6, 10, 12, 2], ["Constitution Mod", 3, 5, 7], ["Tough", 0, 0, 1],
    #                             ["Durable", 0, 0, 1], ["Periapt", 0, 0, 1]], y_lim="max")
    # alt_fighter_greatsword_final()

    # dmg = dnd.hit(d20.adv(), 5 + 6 + 3 - 5, 18).switch(0, d8.reroll_on(1) ** 2 + 20, d8.reroll_on(1) ** 4 + 20).print_normal()
    # spell_hit_die()

    # pf.check(d20, 0, 5).print_normal()
    # pf2e_map()

    # (d6**d4).print_normal()

    # pf2e_treat_wounds()

    # dice.plot_mean(pf2e_scout, ["Bonus", 0, 0, 20])
    pf2e_hunted_shot_vs_hunters_aim()

    # dice.plot_mean([lambda bonus: (pf.check(d20, bonus, 15) == 0) << "Crit Fail",
    #                 lambda bonus: (pf.check(d20, bonus, 15) == 1) << "Fail",
    #                 lambda bonus: (pf.check(d20, bonus, 15) == 2) << "Success",
    #                 lambda bonus: (pf.check(d20, bonus, 15) == 3) << "Crit Success"],
    #                ["Check Bonus", 0, 0, 25])