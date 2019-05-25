# Tudor Berariu, 2016

from random import choice

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "STAY")
ACTION_EFFECTS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "STAY": (0, 0)
}

MOVE_REWARD = -0.1
WIN_REWARD = 10.0
LOSE_REWARD = -10.0


class Game(object):
    def __init__(self, map_file_path, all_actions_legal=True):
        with open(map_file_path) as map_file:
            self._map = [list(line.strip()) for line in map_file]

        self._g_pos = self._get_position("G")
        self._b_pos = self._get_position("B")
        self._o_pos = self._get_position("o")
        self._score = 0

        if all_actions_legal:
            self._legal_actions = [
                [ACTIONS for g_col in range(len(self._map[g_row]))]
                for g_row in range(len(self._map))
            ]
        else:
            self._legal_actions = [
                [
                    tuple(
                        action for action in ACTIONS if self._is_valid_cell(
                            g_row + ACTION_EFFECTS[action][0],
                            g_col + ACTION_EFFECTS[action][1]
                        )
                    ) for g_col in range(len(self._map[g_row]))
                ]
                for g_row in range(len(self._map))
            ]

    def _get_position(self, marker):
        for row_idx, row in enumerate(self._map):
            for col_idx, elem in enumerate(row):
                if marker == elem:
                    return row_idx, col_idx
        return -1, -1

    # Check if the given coordinates are valid (on map and not a wall)
    def _is_valid_cell(self, row, col):
        return (
                row >= 0 and row < len(self._map) and
                col >= 0 and col < len(self._map[row]) and
                self._map[row][col] != "*"
        )

    def _get_enemy_move(self):
        b_row, b_col = self._b_pos
        # The balaur must be on the map
        assert b_row >= 0 and b_col >= 0

        g_row, g_col = self._g_pos
        d_y, d_x = g_row - b_row, g_col - b_col

        is_good = lambda dr, dc: self._is_valid_cell(b_row + dr, b_col + dc)

        next_b_row, next_b_col = b_row, b_col
        if abs(d_y) > abs(d_x) and is_good(int(d_y / abs(d_y)), 0):
            next_b_row = b_row + int(d_y / abs(d_y))
        elif abs(d_x) > abs(d_y) and is_good(0, int(d_x / abs(d_x))):
            next_b_col = b_col + int(d_x / abs(d_x))
        else:
            options = []
            if abs(d_x) > 0:
                if is_good(0, int(d_x / abs(d_x))):
                    options.append((b_row, b_col + int(d_x / abs(d_x))))
            else:
                if is_good(0, -1):
                    options.append((b_row, b_col - 1))
                if is_good(0, 1):
                    options.append((b_row, b_col + 1))
            if abs(d_y) > 0:
                if is_good(int(d_y / abs(d_y)), 0):
                    options.append((b_row + int(d_y / abs(d_y)), b_col))
            else:
                if is_good(-1, 0):
                    options.append((b_row - 1, b_col))
                if is_good(1, 0):
                    options.append((b_row + 1, b_col))

            if options:
                next_b_row, next_b_col = choice(options)

        return next_b_row, next_b_col

    def _apply_action(self, action):
        assert action in ACTIONS
        message = "Greuceanu moved %s." % action

        state = self._map
        g_row, g_col = self._g_pos
        # Greuceanu must be on the map
        assert g_row >= 0 and g_col >= 0
        state[g_row][g_col] = " "

        next_g_row = g_row + ACTION_EFFECTS[action][0]
        next_g_col = g_col + ACTION_EFFECTS[action][1]

        if not self._is_valid_cell(next_g_row, next_g_col):
            next_g_row = g_row
            next_g_col = g_col
            message += " Not a valid cell there."

        # Update Greuceanu's position
        state[next_g_row][next_g_col] = "G"
        self._g_pos = next_g_row, next_g_col

        if self._g_pos == self._b_pos:
            message += " Greuceanu stepped on the balaur."
            return LOSE_REWARD, message
        elif self._g_pos == self._o_pos:
            message += " Greuceanu found 'marul fermecat'."
            return WIN_REWARD, message

        # Balaur moves now
        b_row, b_col = self._b_pos
        state[b_row][b_col] = " "

        next_b_row, next_b_col = self._get_enemy_move()
        state[next_b_row][next_b_col] = "B"
        self._b_pos = next_b_row, next_b_col

        if self._b_pos == self._g_pos:
            message += " The balaur ate Greuceanu."
            reward = LOSE_REWARD
        elif self._b_pos == self._o_pos:
            message += " The balaur found marul fermecat. Greuceanu lost!"
            reward = LOSE_REWARD
        else:
            message += " The balaur follows Greuceanu."
            reward = MOVE_REWARD

        return reward, message

    def apply_action(self, action):
        reward, message = self._apply_action(action)

        self._score += reward

        return reward, message

    # Check if the game is over
    def is_over(self):
        return (
            # Too many moves
                self._score < -20.0
                # LOSS - Greuceanu stepped on the balaur or the balaur ate Greuceanu
                or self._g_pos == self._b_pos
                # WIN - Greuceanu found 'marul fermecat'
                or self._g_pos == self._o_pos
                # LOSS - The balaur found 'marul fermecat'
                or self._b_pos == self._o_pos
        )

    @property
    def state(self):
        return "\n".join("".join(line) for line in self._map)

    @property
    def legal_actions(self):
        g_row, g_col = self._g_pos

        # Legal actions are precomputed for fast retrieval
        return self._legal_actions[g_row][g_col]

    @property
    def score(self):
        return self._score
