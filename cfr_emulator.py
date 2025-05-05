from pypokerengine.api.emulator import Emulator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.data_encoder import DataEncoder
from cfr_utils import get_info_set_key
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json

class Node:
    def __init__(self, bet_options):
        self.num_actions = bet_options
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)

# not -- actions are indexed by fold, all  ---- 0, 1, 2
    def get_strategy(self):
        normalizing_sum = 0

        # CFR+ logic: only positive regrets are used
        positive_regret_sum = np.maximum(0, self.regret_sum)
        normalizing_sum = np.sum(positive_regret_sum)

        if normalizing_sum > 0:
            self.strategy = positive_regret_sum / normalizing_sum
        else:
            # Default to uniform random strategy if all regrets are non-positive
            self.strategy = np.ones(self.num_actions) / self.num_actions

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = np.zeros(self.num_actions)
        normalizing_sum = np.sum(self.strategy_sum)

        
        if normalizing_sum > 0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            avg_strategy = np.ones(self.num_actions) / self.num_actions

        # save as list to put in json
        return avg_strategy.tolist()



class CFR(object):
    def __init__(self, player1, player2):
        self.INITIAL_STACK = 1000
        self.TOTAL_STACK = 2000
        self.TOTAL_BLINDS = 30
        MAX_ROUNDS = 500

        # Set up emulator
        self.em = Emulator()
        self.em.set_game_rule(2, MAX_ROUNDS, 10, 0)
        self.em.register_player("p1", player1)
        self.em.register_player("p2", player2)
        players_info = {
            "p1": { "name": "player1", "stack": self.INITIAL_STACK },
            "p2": { "name": "player2", "stack": self.INITIAL_STACK },
        }

        self.players = [player1, player2]
        self.nodes = {}

        self.iterations = 0
        self.initial_state = self.em.generate_initial_game_state(players_info)
        self.encoder = DataEncoder()

        # for plotting
        self.nodes_touched = 0
        self.total_calls = 0



    def run(self, iterations, load_fpath=None):
        # Load previouslt saved nodes from file if specified
        if load_fpath is not None:
            with open(load_fpath, "rb") as f:
                self.nodes = pickle.load(f)

        nodes_by_round = []
        total_calls_by_round = []
        if load_fpath is not None:
            with open("saves/data.json", "r") as f:
                data = json.load(f)
                nodes_by_round = data["unique_nodes"]
                total_calls_by_round = data["total_calls"]
                self.nodes_touched = nodes_by_round[-1]
                self.total_calls = total_calls_by_round[-1]

        for i in range(iterations):
            # for new run
            i = i + 2900
            self.cfr()
            self.iterations += 1
            nodes_by_round.append(self.nodes_touched)
            total_calls_by_round.append(self.total_calls)
            
            # save the nodes as pkl
            if i % 100 == 99:
                self.save_nodes(f"saves/nodes.pkl")
                self.save_strategy(f"saves/strategy_{i+1}.json")
                self.save_data(f"saves/data.json", nodes_by_round, total_calls_by_round)
                print(f"Saved nodes, strategy, and data to file -- trained for {i+1} iterations", flush=True)

        # final saves
        self.save_nodes(f"saves/nodes_final_{self.iterations}.pkl")
        self.save_strategy(f"saves/strategy_final_{self.iterations}.json")
        self.save_data(f"saves/data_final_{self.iterations}.json", nodes_by_round, total_calls_by_round)
        print(f"Final save complete after {self.iterations} iterations.")

        # return lists for plotting
        return nodes_by_round, total_calls_by_round
                
                
    # start cfr for a fresh round
    def cfr(self):
        # alternate forst action
        if self.iterations % 2 == 1:
            self.initial_state['table'].shift_dealer_btn()

        cfr_state, _ = self.em.start_new_round(self.initial_state)
        p1_stack = np.random.randint(0, (self.TOTAL_STACK - self.TOTAL_BLINDS) + 1)
        p2_stack = (self.TOTAL_STACK - self.TOTAL_BLINDS) - p1_stack
        cfr_state['table'].seats.players[0].stack = p1_stack
        cfr_state['table'].seats.players[1].stack = p2_stack
             
        # set initial prob, stacks, and player
        reach_probs = np.ones(2)
        initial_stacks = [p1_stack, p2_stack]
        start_player = cfr_state['next_player']
        # run cfr
        self.cfr_helper(cfr_state, reach_probs, start_player, initial_stacks)


    # helper function -- CFR logic
    def cfr_helper(self, state, reach_prob, player, initial_stacks):
        # Terminal round node
        if state['street'] == Const.Street.FINISHED:
            curr_stack = state['table'].seats.players[player].stack
            payoff = curr_stack - initial_stacks[player]
            return payoff

        # Non-terminal (action) node
        self.total_calls += 1

         # get state infortmation
        round_state = self.encoder.encode_round_state(state)
        available_actions = self.em.generate_possible_actions(state)
        num_actions = len(available_actions)
        player_uuid = self.players[player].uuid
        hole = state['table'].seats.players[player].hole_card
        info_set_key = get_info_set_key(hole, round_state, num_actions, player_uuid)
        
        # create a new node if one doesnt exist
        if info_set_key not in self.nodes:
            node = Node(num_actions)
            self.nodes[info_set_key] = node
            self.nodes_touched += 1
        else:
            node = self.nodes[info_set_key]

        # CFR logic
        strategy = node.get_strategy()
        utils = np.zeros(num_actions)
        node_util = 0.0
        opponent = (player + 1) % 2
        for a, action_info in enumerate(available_actions):
            prob = strategy[a]
            action = action_info['action']
            new_state, _ = self.em.apply_action(state, action)
            
            new_reach_prob = reach_prob.copy()
            new_reach_prob[player] *= prob
            sampled_util = -self.cfr_helper(new_state, new_reach_prob, opponent, initial_stacks)
            utils[a] = sampled_util
            node_util += prob * sampled_util
        
        for a in range(num_actions):
            regret = utils[a] - node_util
            node.regret_sum[a] += (regret * reach_prob[opponent])

            node.strategy_sum[a] += (reach_prob[player] * strategy[a])

        return node_util
    

    # save the nodes to a file (done intermittently throughout training)
    def save_nodes(self, fpath):
        # save the nodes to a file
        with open(fpath, "wb") as f:
            pickle.dump(self.nodes, f)


    def save_data(self, outpath, nodes_touched, total_calls):
        # save the nodes to a file
        data = {"unique_nodes": nodes_touched, "total_calls": total_calls}
        with open(outpath, "w") as f:
            json.dump(data, f)
	

    # get the average strategy for each node and save to json
    def save_strategy(self, outpath, load_fpath=None):
        # map all all nodes in info_set to theri average strategy and return the resulting dict
        strategy = {}
        if load_fpath is not None:
            with open(load_fpath, "rb") as f:
                nodes = pickle.load(f)
        else:
            nodes = self.nodes

        for key, node in nodes.items():
            strategy[key] = node.get_average_strategy()

        # save to file
        with open(outpath, "w") as f:
            json.dump(strategy, f)


if __name__ == "__main__":
    NUM_ITERS = 10000000
    player1 = BasePokerPlayer()
    player1.set_uuid("p1")
    player2 = BasePokerPlayer()
    player2.set_uuid("p2")

    cfr = CFR(player1, player2)
    nodes, calls = cfr.run(NUM_ITERS, load_fpath="saves/nodes.pkl")
    cfr.save_strategy(f"saves/final_strategy_{NUM_ITERS}.json")

    # Plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].plot(nodes)
    ax[0].set_xlabel("Rounds")
    ax[0].set_ylabel("Unique Nodes Explored")

    ax[1].plot(calls)
    ax[1].set_xlabel("Rounds")
    ax[1].set_ylabel("Total Node Explorations")

    plt.tight_layout()
    plt.savefig("cfr_results.png")
