from production import forward_chain, backward_chain
from utils import print_element_with_capital
from rules import FANTASY_RULES
import random
import re


def tuple_to_string(obj_tuple, delimiter=', '):
    result_string = ''
    for obj in obj_tuple:
        result_string += str(obj) + delimiter
    if result_string:
        result_string = result_string[:-len(delimiter)]
    return result_string


def define_questions_from_goal_tree(FANTASY_RULES):
    # retrieve all final or intermediate truths
    pattern = r"THEN\('(.*?)'\)"
    intermediate_and_final_truths = re.findall(pattern, tuple_to_string(FANTASY_RULES))

    # retrieve all antecedents before THEN
    all_truths = []
    pattern = r"(?<=AND\()'(.*?)'(?=\))|(?<=OR\()'(.*?)'(?=\))"
    matches = re.findall(pattern, tuple_to_string(FANTASY_RULES))
    conditions = [condition for match in matches for condition in match if condition]
    for condition in conditions:
        input_string = condition.replace("'", "")
        elements = input_string.split(",")
        for element in elements:
            all_truths.append(element.strip())

    # retrieve all initial truths
    initial_truths = [item for item in all_truths if item not in intermediate_and_final_truths]
    mapped_objects = {item.replace('(?x)', 'He') + '?': item for item in initial_truths}
    initial_truth_questions_map = {value.replace('(?x) ', ''): key for key, value in mapped_objects.items()}

    # filter intermediate truths
    intermediate_truths = list(set(all_truths).intersection(intermediate_and_final_truths))
    for inter in intermediate_truths:
        for intermediate in intermediate_truths:
            intermediate_list = backward_chain(FANTASY_RULES, intermediate)
            intermediate_list = [item for item in intermediate_list if item != intermediate]
            if any('OR' in f' {item} ' for item in intermediate_list):
                intermediate_truths.remove(intermediate)
            if any(item in intermediate_truths for item in intermediate_list):
                intermediate_truths.remove(intermediate)
    # now filter intermediate that have common initial truths
    for intermediate_extern in intermediate_truths:
        intermediate_extern_list = backward_chain(FANTASY_RULES, intermediate_extern)
        intermediate_extern_list = [item for item in intermediate_extern_list if item != intermediate_extern]
        for intermediate_intern in intermediate_truths:
            if intermediate_extern != intermediate_intern:
                intermediate_intern_list = backward_chain(FANTASY_RULES, intermediate_intern)
                intermediate_intern_list = [item for item in intermediate_intern_list if item != intermediate_intern]
                common_items = list(set(intermediate_extern_list).intersection(intermediate_intern_list))
                if len(common_items) > 0:
                    intermediate_truths.remove(intermediate_extern)
                    intermediate_truths.remove(intermediate_intern)

    return initial_truth_questions_map, intermediate_truths


def generate_question(INITIAL_TRUTHS):
    random_key = random.choice(list(INITIAL_TRUTHS.keys()))
    random_value = INITIAL_TRUTHS[random_key]
    return random_key, random_value


def remove_question_from_map(INITIAL_TRUTHS, truth):
    if truth in INITIAL_TRUTHS:
        INITIAL_TRUTHS.pop(truth)


def remove_unnecessary_truths(INITIAL_TRUTHS, MID_AND_TRUTHS, truth):
    check_truth = "he " + truth
    for mid_truth in MID_AND_TRUTHS:
        list_bounded_truths = backward_chain(FANTASY_RULES, "he " + mid_truth)
        if check_truth in list_bounded_truths:
            for elem in list_bounded_truths:
                trimmed_elem = elem.replace("he", "").strip()
                remove_question_from_map(INITIAL_TRUTHS, trimmed_elem)


def akinator_clone_algorithm():
    INITIAL_TRUTHS, MID_AND_TRUTHS = define_questions_from_goal_tree(FANTASY_RULES)

    character_info = ()
    result = ""
    while len(INITIAL_TRUTHS) > 0:
        truth, question = generate_question(INITIAL_TRUTHS)
        answer = input(question + "  (Y/N):  ")
        if answer.lower() == "y":
            character_info += ("he " + truth,)
        elif answer.lower() == "n":
            remove_unnecessary_truths(INITIAL_TRUTHS, MID_AND_TRUTHS, truth)
        else:
            continue
        remove_question_from_map(INITIAL_TRUTHS, truth)
        if print_element_with_capital(forward_chain(FANTASY_RULES, character_info)) is not None:
            result = print_element_with_capital(forward_chain(FANTASY_RULES, character_info))
            break
    if len(result) == 0:
        print("Answer -> You are watching an ordinary city folk")
    else:
        print("Answer # " + result)


if __name__ == '__main__':
    # Test forward chaining algo for predefined
    print_element_with_capital(forward_chain(FANTASY_RULES, BARD_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, KNIGHT_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, MONK_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, ALCHEMIST_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, ARCHMAGE_DATA))

    # Test backward chaining
    print(backward_chain(FANTASY_RULES, 'bodhidharma is handsome'))
    print(backward_chain(FANTASY_RULES, 'bodhidharma is a MONK'))
    print(backward_chain(FANTASY_RULES, 'bodhidharma is a BARD'))

    # Test generation of questions from initial map
    print(generate_question())

    # Start expert system
    akinator_clone_algorithm()
