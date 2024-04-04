from production import forward_chain
from utils import print_element_with_capital
from rules import FANTASY_RULES
from rules import BARD_DATA
from rules import KNIGHT_DATA
from rules import MONK_DATA
from rules import ALCHEMIST_DATA
from rules import ARCHMAGE_DATA

if __name__ == '__main__':
    print_element_with_capital(forward_chain(FANTASY_RULES, BARD_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, KNIGHT_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, MONK_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, ALCHEMIST_DATA))
    print_element_with_capital(forward_chain(FANTASY_RULES, ARCHMAGE_DATA))