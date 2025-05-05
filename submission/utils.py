from pypokerengine.utils.card_utils import *
from time import time
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

    spr_cat = get_bucket(SPR, [1, 2, 5, 10, 20])

    return spr_cat, is_shorter_stack


def get_info_set_key(hole, state, legal_actions, uuid):

    # street
    street = state["street"]
    if street not in ["preflop", "flop", "turn", "river", "showdown", "finished"]:
        raise ValueError(f"Invalid street: {street}")
    street_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4, 'finished': 5}
    street = street_mapping.get(street)

    # win rate
    community = state["community_card"]
    win_rate = estimate_hole_card_win_rate(100, 2, gen_cards(hole), gen_cards(community))
    win_rate_bucket = get_bucket(win_rate, [0.2, 0.4, 0.6, 0.8])

    # texture
    texture = classify_board(community)

    # stack info
    spr_cat, is_shorter_stack = pot_stack_values(state, uuid)

    # raise allowed flag
    raise_allowed = int(legal_actions > 2)

    # TODO - incorporate action history?

    key = f"{street}{texture}{win_rate_bucket}{spr_cat}{is_shorter_stack}{raise_allowed}"

    # NOTE: there are up to 5 * 5 * 5 * 6 * 2 * 2 = 3000 information sets 
    return key


def load_strategy(fpath):
    with open(fpath, 'r') as f:
        strategy = json.load(f)
    return strategy