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
    for mid_truth in MID_AND_TRUTHS:
        list_bounded_truths = backward_chain(FANTASY_RULES, "he " + mid_truth)
        if truth in list_bounded_truths:
            for elem in list_bounded_truths:
                print(elem)
                trimmed_elem = elem.replace("he", "").strip()
                print(trimmed_elem)
                if trimmed_elem in INITIAL_TRUTHS:
                    INITIAL_TRUTHS.pop(trimmed_elem)


def akinator_clone_algorithm():
    character_info = ()
    while len(INITIAL_TRUTHS) > 0:
        truth, question = generate_question()
        answer = input(question + "  (Y/N):  ")
        if answer.lower() == "y":
            character_info += ("he " + truth,)
        else:
            remove_unnecessary_truths(truth)
        INITIAL_TRUTHS.pop(truth)
        if print_element_with_capital(forward_chain(FANTASY_RULES, character_info)) is not None:
            print_element_with_capital(forward_chain(FANTASY_RULES, character_info))
            break
    print(character_info)
    print(forward_chain(FANTASY_RULES, character_info))


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

    # akinator_clone_algorithm()

    remove_unnecessary_truths('he has blue eyes')
