# Tudor Berariu, 2016
# Razvan Chitu, 2018

import math

# Standard library imports
from argparse import ArgumentParser
from random import choice, random
from time import sleep

import numpy as np

# External library imports
from matplotlib import pyplot as plt

# Local imports
from mini_pacman import Game


def epsilon_greedy(Q, state, legal_actions, epsilon):
    # TODO (2) : Epsilon greedy
    unexplored_actions = []
    for action in legal_actions:
        if (state, action) not in Q:
            unexplored_actions.append(action)

    if len(unexplored_actions) == 0:
        return np.random.choice(
            a=[best_action(Q, state, legal_actions), choice(legal_actions)], p=[1 - epsilon, epsilon]
        )
    else:
        return choice(unexplored_actions)


def best_action(Q, state, legal_actions):
    q_max = -math.inf
    a_star = None
    for action in legal_actions:
        q = Q.get((state, action), 0)
        if q > q_max:
            q_max = q
            a_star = action
    return a_star


def q_learning(
    map_file,
    learning_rate,
    discount,
    epsilon,
    train_episodes,
    eval_every,
    eval_episodes,
    verbose,
    plot_scores,
    sleep_interval,
    final_show,
):
    # Q will use (state, action) tuples as key.
    # Use Q.get(..., 0) for default values.
    Q = {}
    train_scores = []
    eval_scores = []

    # for each episode ...
    for train_ep in range(1, train_episodes + 1):
        game = Game(map_file, all_actions_legal=False)

        # display current state and sleep
        if verbose:
            print(game.state)
            sleep(sleep_interval)

        # while current state is not terminal
        while not game.is_over():
            # choose one of the legal actions
            state, actions = game.state, game.legal_actions
            action = epsilon_greedy(Q, state, actions, epsilon)

            # apply action and get the next state and the reward
            reward, msg = game.apply_action(action)
            next_state, next_actions = game.state, game.legal_actions

            # TODO (1) : Q-Learning

            max_q = -math.inf
            a_star = None
            for next_action in next_actions:
                q = Q.get((next_state, next_action), 0)
                if q > max_q:
                    max_q = q
                    a_star = next_action

            Q[(state, action)] = Q.get((state, action), 0) + learning_rate * (
                reward + discount * Q.get((next_state, a_star), 0) - Q.get((state, action), 0)
            )

            # display current state and sleep
            if verbose:
                print(msg)
                print(game.state)
                sleep(sleep_interval)

        print("Episode %6d / %6d" % (train_ep, train_episodes))
        train_scores.append(game.score)

        # evaluate the greedy policy
        if train_ep % eval_every == 0:
            avg_score = 0.0

            # TODO (4) : Evaluate
            game = Game(map_file, all_actions_legal=False)
            while not game.is_over():
                state, actions = game.state, game.legal_actions
                action = best_action(Q, state, actions)
                reward, msg = game.apply_action(action)
                avg_score += reward
            eval_scores.append(avg_score)

    # --------------------------------------------------------------------------
    if final_show:
        game = Game(map_file, all_actions_legal=False)
        while not game.is_over():
            state, actions = game.state, game.legal_actions
            action = best_action(Q, state, actions)
            reward, msg = game.apply_action(action)
            print(msg)
            print(game.state)
            sleep(sleep_interval)

    if plot_scores:
        plt.xlabel("Episode")
        plt.ylabel("Average score")
        plt.plot(
            np.linspace(1, train_episodes, train_episodes),
            np.convolve(train_scores, [0.2, 0.2, 0.2, 0.2, 0.2], "same"),
            linewidth=1.0,
            color="blue",
        )
        plt.plot(np.linspace(eval_every, train_episodes, len(eval_scores)), eval_scores, linewidth=2.0, color="red")
        plt.show()


def main():
    parser = ArgumentParser()
    # Input file
    parser.add_argument("--map_file", type=str, default="mini_map", help="File to read map from.")
    # Meta-parameters
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--discount", type=float, default=0.99, help="Value for the discount factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Probability to choose a random action.")
    # Training and evaluation episodes
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate policy every ... games.")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of games to play for evaluation.")
    # Display
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print each state")
    parser.add_argument(
        "--plot",
        dest="plot_scores",
        action="store_true",
        help="Plot scores in the end",
    )
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to 'sleep' between moves.")
    parser.add_argument("--final_show", dest="final_show", action="store_true", help="Demonstrate final strategy.")
    args = parser.parse_args()

    q_learning(
        args.map_file,
        args.learning_rate,
        args.discount,
        args.epsilon,
        args.train_episodes,
        args.eval_every,
        args.eval_episodes,
        args.verbose,
        args.plot_scores,
        args.sleep,
        args.final_show,
    )


if __name__ == "__main__":
    main()
