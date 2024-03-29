#!/usr/bin/env python

# Copyright (c) 2019, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                       Jim Mainprice on Thursday Mars 14 2019

# Modified from AI a Modern Approach (AIMA)
# where POMDP has been removed
# https://github.com/aimacode/aima-python/blob/master/mdp.py

"""Markov Decision Processes (Chapter 17)
First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid. We also represent a policy
as a dictionary of {state: action} pairs, and a Utility function as a
dictionary of {state: number} pairs. We then define the value_iteration
and policy_iteration algorithms."""

from . import common_imports
from utils.misc import vector_add, orientations, turn_right, turn_left
import random


class MDP:

    """A Markov Decision Process,
    defined by an initial state, transition model, and reward function.

    We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(self, init, actlist, terminals,
                 transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""

        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        """
        Return a list of actions that can be performed
        in this state. By default, a fixed list of actions, except
        for terminal states. Override this method if you need to specialize
        by state.
        """

        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None

    def check_consistency(self):

        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(
            self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001


class MDP2(MDP):

    """
    Handles terminal states, and transitions to and
    from terminal states better.
    """

    def __init__(self, init, actlist, terminals,
                 transitions, reward=None, gamma=0.9):
        MDP.__init__(self, init, actlist, terminals,
                     transitions, reward, gamma=gamma)

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return self.transitions[state][action]


class GridMDP(MDP):

    """
    A two-dimensional grid MDP, as in [Figure 17.1].

    All you have to do is specify the grid as a list of lists of rewards;
    use None for an obstacle (unreachable state). Also, you should
    specify the terminal states. An action is an (x, y) unit vector;
    e.g. (1, 0) means move east.
    """

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma)

    def calculate_T(self, state, action):
        if action:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0): '>', (0, 1): '^',
                 (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})



""" [Figure 17.1]
A 4x3 grid environment that presents the agent
with a sequential decision problem.
"""

sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])


__doc__ += """
>>> pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))
>>> sequential_decision_environment.to_arrows(pi)
[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '>', '^', '<']]
>>> from utils import print_table
>>> print_table(sequential_decision_environment.to_arrows(pi))
>   >      >   .
^   None   ^   .
^   >      ^   <
>>> print_table(sequential_decision_environment.to_arrows(policy_iteration(sequential_decision_environment)))
>   >      >   .
^   None   ^   .
^   >      ^   <
"""  # noqa

"""
s = { 'a' : {   'plan1' : [(0.2, 'a'), (0.3, 'b'), (0.3, 'c'), (0.2, 'd')],
                'plan2' : [(0.4, 'a'), (0.15, 'b'), (0.45, 'c')],
                'plan3' : [(0.2, 'a'), (0.5, 'b'), (0.3, 'c')],
            },
      'b' : {   'plan1' : [(0.2, 'a'), (0.6, 'b'), (0.2, 'c'), (0.1, 'd')],
                'plan2' : [(0.6, 'a'), (0.2, 'b'), (0.1, 'c'), (0.1, 'd')],
                'plan3' : [(0.3, 'a'), (0.3, 'b'), (0.4, 'c')],
            },
      'c' : {   'plan1' : [(0.3, 'a'), (0.5, 'b'), (0.1, 'c'), (0.1, 'd')],
                'plan2' : [(0.5, 'a'), (0.3, 'b'), (0.1, 'c'), (0.1, 'd')],
                'plan3' : [(0.1, 'a'), (0.3, 'b'), (0.1, 'c'), (0.5, 'd')],
            },
    }
"""
