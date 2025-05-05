from pypokerengine.utils.card_utils import *
import json


def get_bucket(value, thresholds):
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    return len(thresholds)


def classify_board(community_cards):
    if len(community_cards) == 0:
        return 0
    
    ranks = [card[1] for card in community_cards]
    suits = [card[0] for card in community_cards]

    flush_draw = any(suits.count(suit) >= 3 for suit in suits)
    paired = len(set(ranks)) < len(ranks)
    high_card = max(ranks)

    if paired:
        return 1
    if flush_draw:
        return 2
    
    # dry: return 3
    return 3

def pot_stack_values(state, uuid):
    seats = state["seats"]
    my_stack = 0
    other_stack = 0
    
    for seat in seats:
        if seat["uuid"] == uuid:
            my_stack = seat["stack"]
        else:
            other_stack = seat["stack"]
    
    pot = state["pot"]["main"]["amount"]

    if my_stack < other_stack:
        effective_stack = my_stack
        is_shorter_stack = 1
    else:
        effective_stack = other_stack
        is_shorter_stack = 0
    
    SPR = effective_stack / pot if pot > 0 else float("inf")

    spr_cat = get_bucket(SPR, [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10])

    return spr_cat, is_shorter_stack


def get_info_set_key(hole, state, legal_actions, uuid):

    # street
    street = state["street"]
    street_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4, 'finished': 5}
    street = street_mapping.get(street)

    # win rate
    community = state["community_card"]

    # NOTE: change to gen_cards(hole) for the final version
    win_rate = estimate_hole_card_win_rate(100, 2, hole, gen_cards(community))
    win_rate_bucket = get_bucket(win_rate, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # texture
    texture = classify_board(community)

    # stack info
    spr_cat, is_shorter_stack = pot_stack_values(state, uuid)

    # raise allowed flag
    raise_allowed = int(legal_actions > 2)

    key = f"{street}{texture}{win_rate_bucket}{spr_cat}{is_shorter_stack}{raise_allowed}"

    # NOTE: there are up to 5 * 3 * 10 * 10 * 2 * 2 = 6000 unique information sets
    ## in practice there are fewer than 6000 because some keys combinations are not possible
    return key


def load_strategy(fpath):
    with open(fpath, 'r') as f:
        strategy = json.load(f)
    return strategy