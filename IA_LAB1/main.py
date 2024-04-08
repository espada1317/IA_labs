from production import forward_chain, backward_chain
from utils import print_element_with_capital, printTrues
from rules import FANTASY_RULES, INITIAL_TRUTHS, MID_AND_TRUTHS
from rules import BARD_DATA
from rules import KNIGHT_DATA
from rules import MONK_DATA
from rules import ALCHEMIST_DATA
from rules import ARCHMAGE_DATA
import random


def generate_question():
    random_key = random.choice(list(INITIAL_TRUTHS.keys()))
    random_value = INITIAL_TRUTHS[random_key]
    return random_key, random_value


def remove_unnecessary_truths(truth):
    check_truth = "he " + truth
    for mid_truth in MID_AND_TRUTHS:
        list_bounded_truths = backward_chain(FANTASY_RULES, "he " + mid_truth)
        if check_truth in list_bounded_truths:
            for elem in list_bounded_truths:
                trimmed_elem = elem.replace("he", "").strip()
                if trimmed_elem in INITIAL_TRUTHS:
                    INITIAL_TRUTHS.pop(trimmed_elem)


def akinator_clone_algorithm():
    character_info = ()
    result = ""
    while len(INITIAL_TRUTHS) > 0:
        truth, question = generate_question()
        answer = input(question + "  (Y/N):  ")
        if answer.lower() == "y":
            character_info += ("he " + truth,)
        elif answer.lower() == "n":
            remove_unnecessary_truths(truth)
        else:
            continue
        if truth in INITIAL_TRUTHS:
            INITIAL_TRUTHS.pop(truth)
        if (print_element_with_capital(forward_chain(FANTASY_RULES, character_info)) is not None
                and len(print_element_with_capital(forward_chain(FANTASY_RULES, character_info))) > 0):
            result = print_element_with_capital(forward_chain(FANTASY_RULES, character_info))
            break
    if len(result) == 0:
        print("Answer -> You are watching an ordinary city folk")
    else:
        print("Answer # " + result)


if __name__ == '__main__':
    # print_element_with_capital(forward_chain(FANTASY_RULES, BARD_DATA))
    # print_element_with_capital(forward_chain(FANTASY_RULES, KNIGHT_DATA))
    # print_element_with_capital(forward_chain(FANTASY_RULES, MONK_DATA))
    # print_element_with_capital(forward_chain(FANTASY_RULES, ALCHEMIST_DATA))
    # print_element_with_capital(forward_chain(FANTASY_RULES, ARCHMAGE_DATA))

    # print("\nBackward chaining:")
    # printTrues(backward_chain(FANTASY_RULES, 'taliesin is a BARD'))
    # printTrues(backward_chain(FANTASY_RULES, 'bodhidharma is a MONK'))

    # print(generateQuestion(INITIAL_TRUTHS))
    # print(generateQuestion(INITIAL_TRUTHS))

    akinator_clone_algorithm()

    # remove_unnecessary_truths('he has blue eyes')
