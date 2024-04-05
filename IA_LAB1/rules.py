from production import IF, AND, THEN, OR, DELETE, NOT, FAIL

# MEDIEVAL FANTASY CHARACTERS RULES
FANTASY_RULES = (

    IF(AND('(?x) has bird feather',  # R1
           '(?x) has musical instrument'),
       THEN('(?x) is a poet')),

    IF(AND('(?x) has colorful clothes',  # R2
           '(?x) has jewelry'),
       THEN('(?x) is stylish')),

    IF(AND('(?x) wears sword',  # R3
           '(?x) wears shining armor'),
       THEN('(?x) is warrior')),

    IF(AND('(?x) has blonde hair',  # R4
           '(?x) has blue eyes',
           '(?x) is young'),
       THEN('(?x) is handsome')),

    IF(OR('(?x) travels on horse',  # R5
          '(?x) travels lightly'),
       THEN('(?x) is adventurer')),

    IF(AND('(?x) wears robes',  # R7
           '(?x) wears black clothing'),
       THEN('(?x) has symbolic profession')),

    IF(AND('(?x) is bald',  # R8
           '(?x) has beads',
           '(?x) has beard',
           '(?x) is old'),
       THEN('(?x) is believer')),

    IF(OR('(?x) has symbol of cross',   # R9
          "(?x) is believer"),
       THEN('(?x) is christian')),

    IF(AND('(?x) has beard',  # R10
           '(?x) is old',
           '(?x) has grizzle hair'),
       THEN('(?x) has a lot of experience')),

    IF(AND('(?x) has a lot of experience',  # R11
           '(?x) has mysterious books',
           '(?x) has potions'),
       THEN('(?x) studies science')),

    IF(AND('(?x) studies science',  # R12
           '(?x) has runes'),
       THEN('(?x) studies magical science')),

    IF(AND('(?x) studies magical science',  # 13
           '(?x) has symbolic profession',
           '(?x) has magic staff'),
       THEN('(?x) is an ARCHMAGE')),

    IF(AND('(?x) has symbolic profession',  # R14
           '(?x) has star sign',
           '(?x) studies magical science'),
       THEN('(?x) is an ALCHEMIST')),

    IF(AND('(?x) is adventurer',  # R15
           '(?x) has symbolic profession',
           '(?x) is christian'),
       THEN('(?x) is a MONK')),

    IF(AND('(?x) has herald',  # R16
           '(?x) is warrior',
           '(?x) is handsome',
           '(?x) is adventurer',
           '(?x) is christian'),
       THEN('(?x) is a KNIGHT')),

    IF(AND('(?x) is a poet',  # R17
           '(?x) is stylish',
           '(?x) is handsome',
           '(?x) is adventurer'),
       THEN('(?x) is a BARD')),

)

BARD_DATA = (
    'taliesin has bird feather',
    'taliesin has musical instrument',
    'taliesin has colorful clothes',
    'taliesin has jewelry',
    'taliesin has blonde hair',
    'taliesin has blue eyes',
    'taliesin is young',
    'taliesin travels on horse',
)

KNIGHT_DATA = (
    'archibald has herald',
    'archibald wears sword',
    'archibald wears shining armor',
    'archibald has blonde hair',
    'archibald has blue eyes',
    'archibald is young',
    'archibald travels on horse',
    'archibald has symbol of cross',
)

MONK_DATA = (
    'bodhidharma travels lightly',
    'bodhidharma wears robes',
    'bodhidharma wears black clothing',
    'bodhidharma is bald',
    'bodhidharma has beads',
    'bodhidharma has beard',
    'bodhidharma is old',
)

ALCHEMIST_DATA = (
    'nicolas wears robes',
    'nicolas wears black clothing',
    'nicolas has star sign',
    'nicolas has beard',
    'nicolas is old',
    'nicolas has grizzle hair',
    'nicolas has mysterious books',
    'nicolas has potions',
    'nicolas has runes',
)

ARCHMAGE_DATA = (
    'hayul wears robes',
    'hayul wears black clothing',
    'hayul has beard',
    'hayul is old',
    'hayul has grizzle hair',
    'hayul has mysterious books',
    'hayul has potions',
    'hayul has runes',
    'hayul has magic staff',
)
