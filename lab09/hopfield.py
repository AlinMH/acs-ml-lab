import random

import numpy
from PIL import Image


class HopfieldNetwork:
    CHAR_TO_INT = {"_": -1, "X": 1}
    INT_TO_CHAR = {-1: "_", 1: "X"}

    # Initalize a Hopfield Network with N neurons
    def __init__(self, neurons_no):
        self.neurons_no = neurons_no
        self.state = numpy.ones((self.neurons_no), dtype=int)
        self.weights = numpy.zeros((self.neurons_no, self.neurons_no))
        self.last_energy = None

    # ------------------------------------------------------------------------------

    # Learn some patterns
    def learn_patterns(self, patterns, learning_rate):
        # TASK 1
        for _pattern in patterns:
            pattern = list(map(lambda c: HopfieldNetwork.CHAR_TO_INT[c], _pattern))
            self.weights = numpy.add(self.weights, numpy.outer(pattern, pattern))
        self.weights -= len(patterns) * numpy.identity(self.neurons_no)
        print(self.weights)

    # Compute the energy of the current configuration
    def energy(self):
        # TASK 1:
        first_two = numpy.dot(self.state, self.weights)
        last = -0.5 * numpy.dot(first_two, self.state)
        return last

    # Update a single random neuron
    # Update a single random neuron
    def single_update(self):
        i = random.randint(0, len(self.state) - 1)

        val = 0
        for j in range(len(self.state)):
            val += self.weights[i, j] * self.state[j]

        self.state[i] = 1 if val > 0 else -1

    # Check if energy is minimal
    def is_energy_minimal(self):
        for i in range(self.neurons_no):
            val = 0.0
            for j in range(len(self.state)):
                val += self.weights[i, j] * self.state[j]
            val = 1 if val > 0 else -1
            if self.state[i] != val:
                return False
        return True

    # --------------------------------------------------------------------------

    # Approximate the distribution of final states
    # starting from @samples_no random states.
    def get_final_states_distribution(self, samples_no=1000):
        # TASK 3
        hist = {}
        for i in range(samples_no):
            self.random_reset()
            while not self.is_energy_minimal():
                self.single_update()
            print('step %d' % i)
            reached_state = self.get_pattern()
            hist[reached_state] = hist.get(reached_state, 0) + 1
        for key in hist:
            hist[key] = float(hist[key]) / samples_no
        return hist

    # -------------------------------------------------------------------------

    # Unlearn some patterns
    def unlearn_patterns(self, patterns, learning_rate):
        # TASK BONUS
        for _pattern in patterns:
            pattern = list(map(lambda c: HopfieldNetwork.CHAR_TO_INT[c], _pattern))
            self.weights -= learning_rate * numpy.outer(pattern, pattern)

    # -------------------------------------------------------------------------

    # Get the pattern of the state as string
    def get_pattern(self):
        return "".join([HopfieldNetwork.INT_TO_CHAR[n] for n in self.state])

    # Reset the state of the Hopfield Network to a given pattern
    def reset(self, pattern):
        assert (len(pattern) == self.neurons_no)
        for i in range(self.neurons_no):
            self.state[i] = HopfieldNetwork.CHAR_TO_INT[pattern[i]]

    # Reset the state of the Hopfield Network to a random pattern
    def random_reset(self):
        for i in range(self.neurons_no):
            self.state[i] = 1 - 2 * numpy.random.randint(0, 2)

    def to_string(self):
        return HopfieldNetwork.state_to_string(self.state)

    @staticmethod
    def state_to_string(state):
        return "".join([HopfieldNetwork.INT_TO_CHAR[c] for c in state])

    @staticmethod
    def state_from_string(str_state):
        return numpy.array([HopfieldNetwork.CHAR_TO_INT[c] for c in str_state])

    # display the current state of the HopfieldNetwork
    def display_as_matrix(self, rows_no, cols_no):
        assert (rows_no * cols_no == self.neurons_no)
        HopfieldNetwork.display_state_as_matrix(self.state, rows_no, cols_no)

    # display the current state of the HopfieldNetwork
    def display_as_image(self, rows_no, cols_no):
        assert (rows_no * cols_no == self.neurons_no)
        pixels = [1 if s > 0 else 0 for s in self.state]
        img = Image.new('1', (rows_no, cols_no))
        img.putdata(pixels)
        img.show()

    @staticmethod
    def display_state_as_matrix(state, rows_no, cols_no):
        assert (state.size == rows_no * cols_no)
        print("")
        for i in range(rows_no):
            print("".join([HopfieldNetwork.INT_TO_CHAR[state[i * cols_no + j]]
                           for j in range(cols_no)]))
        print("")
