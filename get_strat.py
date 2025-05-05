from cfr_emulator import CFR, Node
from pypokerengine.players import BasePokerPlayer

CFR = CFR(
    BasePokerPlayer(),
    BasePokerPlayer(),
)
CFR.save_strategy("mccfr_strategy.json", load_fpath="saves/nodes.pkl")
