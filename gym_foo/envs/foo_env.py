import random
from collections import deque
import copy

import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np


class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=15, height=15):
        self.state = []
        self.done = 0
        self.add = 0
        self.reward = 0
        self.score = 0
        # TODO changed for softmax!
        # step actions are mapped like this: 0:left, 1:right, 2:straight
        self.action_space = spaces.Discrete(3)
        self.width = width
        self.height = height
        # create set with x-y tuples
        self.empty_cells = {(x, y) for x in range(width) for y in range(height)}
        # initialize coords for snake body in middle of the available field.
        middle_x = int(width / 2)
        middle_y = int(height / 2)
        # TODO Changed!
        # initial_snake_len = 4
        # first_half_snake = int(initial_snake_len / 2)
        # sec_half_snake = initial_snake_len - first_half_snake
        # snake_body = [(middle_x, y) for y in range(middle_y - first_half_snake, middle_y + sec_half_snake)]
        snake_body = [(middle_x, middle_y)]
        self.initial_snake = deque(snake_body)
        self.snake = copy.deepcopy(self.initial_snake)
        for i in self.snake:
            self.empty_cells.remove(i)
        # # create coords for snake head.
        # self.snake_head = (middle_x, middle_y + sec_half_snake)
        self.snake_head_direc = 0
        # self.empty_cells.remove(self.snake_head)
        # create food
        self.dot = random.choice(tuple(self.empty_cells))
        self.empty_cells.remove(self.dot)
        # create mapping for possible actions to an x(0) move and left(-1) or right(1) or an y(1) move and up(1)
        #  or down(-1).
        self.step_direc_map = {
            0: [1, 1],
            1: [0, 1],
            2: [1, -1],
            3: [0, -1]
        }

        # TODO test if state is updated correctly
        # TODO CHANGED!
        # for x, y in self.snake:
        #     self.state[y, x] = 1
        #
        # self.state = np.zeros((height, width))
        # self.state = np.pad(self.state[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), mode='constant',
        #                     constant_values=-1)
        # self.state[self.dot[1], self.dot[0]] = 2

        self.opposite_direc_diff = 2
        self.viewer = None

    def update_direction(self, action):
        if action == 0:
            if self.snake_head_direc == 0:
                self.snake_head_direc += 3
            else:
                self.snake_head_direc -= 1
        if action == 1:
            if self.snake_head_direc == 3:
                self.snake_head_direc -= 3
            else:
                self.snake_head_direc += 1

    #     if action == 2 it means going straight so direction can remain the same.

    # target = action
    def step(self, target):

        # TODO changed for softmax!
        self.update_direction(target)

        if self.done == 1:
            print("Game Over")
            return [self.state, self.reward, self.done, self.add]
        #   check if current action is in the opposite direction of the previous action
        # TODO changed for softmax
        # elif len(self.snake) > 1 and ((target == 0 and self.snake_head_direc == 2) or (target == 1 and self.snake_head_direc == 3) or \
        #         (target == 2 and self.snake_head_direc == 0) or (target == 3 and self.snake_head_direc == 1)):
        #     print("Invalid Step")
        #     return [self.state, self.reward, self.done, self.add]
        else:

            # update snake_head_direc with current action
            # TODO changed for softmax!
            # self.snake_head_direc = target
            old_snake_head = self.snake.pop()
            new_snake_head = list(copy.deepcopy(old_snake_head))
            # TODO changed for softmax!
            # new_snake_head[self.step_direc_map[target][0]] = new_snake_head[self.step_direc_map[target][0]] + \
            #                                                  self.step_direc_map[target][1]
            new_snake_head[self.step_direc_map[self.snake_head_direc][0]] = \
                new_snake_head[self.step_direc_map[self.snake_head_direc][0]] + \
                self.step_direc_map[self.snake_head_direc][1]
            new_snake_head = tuple(new_snake_head)

            if new_snake_head in self.snake:
                print('Your snake hit itself!')
                print("Game over")
                self.done = 1
                self.reward = -10
                # TODO changed ! state should be updated only at end!
                # return [self.state, self.reward, self.done, self.add]
            elif new_snake_head == self.dot:
                print("You ate some food!")
                self.reward = 1
                self.score += 1
                self.snake.append(old_snake_head)
                self.snake.append(new_snake_head)
                self.dot = random.choice(tuple(self.empty_cells))
                self.empty_cells.remove(self.dot)
            #     food was eaten. Snake body increases and popping from the left should not be done.
            elif new_snake_head in self.empty_cells:
                self.snake.append(old_snake_head)
                self.snake.append(new_snake_head)
                self.empty_cells.remove(new_snake_head)
                # food was not eaten. Pop tail
                snake_tail = self.snake.popleft()
                self.empty_cells.add(snake_tail)
                self.reward = -0.1
            else:
                # snake hit the wall
                print('Your snake hit the wall!')
                print("Game over")
                self.done = 1
                self.reward = -10
                # TODO changed ! state should be updated only at end!
                # return [self.state, self.reward, self.done, self.add]

            # TODO changed
            self.state = [
                ((self.snake_head_direc == 0 and ((new_snake_head[0] - 1, new_snake_head[1]) in self.snake or
                                                  ((
                                                       new_snake_head[0] - 1,
                                                       new_snake_head[1]) not in self.empty_cells and
                                                   (new_snake_head[0] - 1, new_snake_head[1]) != self.dot)))
                 or (self.snake_head_direc == 1 and ((new_snake_head[0], new_snake_head[1] + 1) in self.snake or
                                                     ((new_snake_head[0],
                                                       new_snake_head[1] + 1) not in self.empty_cells and
                                                      (new_snake_head[0], new_snake_head[1] + 1) != self.dot)))
                 or (self.snake_head_direc == 2 and ((new_snake_head[0] + 1, new_snake_head[1]) in self.snake or
                                                     ((new_snake_head[0] + 1,
                                                       new_snake_head[1]) not in self.empty_cells and
                                                      (new_snake_head[0] + 1, new_snake_head[1]) != self.dot)))
                 or (self.snake_head_direc == 3 and ((new_snake_head[0], new_snake_head[1] - 1) in self.snake or
                                                     ((new_snake_head[0],
                                                       new_snake_head[1] - 1) not in self.empty_cells and
                                                      (new_snake_head[0], new_snake_head[1] - 1) != self.dot)))),
                # danger left of snake head

                ((self.snake_head_direc == 0 and ((new_snake_head[0], new_snake_head[1] + 1) in self.snake or
                                                  ((
                                                       new_snake_head[0],
                                                       new_snake_head[1] + 1) not in self.empty_cells and
                                                   (new_snake_head[0], new_snake_head[1] + 1) != self.dot)))
                 or (self.snake_head_direc == 1 and ((new_snake_head[0] + 1, new_snake_head[1]) in self.snake or
                                                     ((new_snake_head[0] + 1,
                                                       new_snake_head[1]) not in self.empty_cells and
                                                      (new_snake_head[0] + 1, new_snake_head[1]) != self.dot)))
                 or (self.snake_head_direc == 2 and ((new_snake_head[0], new_snake_head[1] - 1) in self.snake or
                                                     ((new_snake_head[0],
                                                       new_snake_head[1] - 1) not in self.empty_cells and
                                                      (new_snake_head[0], new_snake_head[1] - 1) != self.dot)))
                 or (self.snake_head_direc == 3 and ((new_snake_head[0] - 1, new_snake_head[1]) in self.snake or
                                                     ((new_snake_head[0] - 1,
                                                       new_snake_head[1]) not in self.empty_cells and
                                                      (new_snake_head[0] - 1, new_snake_head[1]) != self.dot)))),
                # danger straight of snake head

                ((self.snake_head_direc == 0 and ((new_snake_head[0] + 1, new_snake_head[1]) in self.snake or
                                                  ((
                                                       new_snake_head[0] + 1,
                                                       new_snake_head[1]) not in self.empty_cells and
                                                   (new_snake_head[0] + 1, new_snake_head[1]) != self.dot)))
                 or (self.snake_head_direc == 1 and ((new_snake_head[0], new_snake_head[1] - 1) in self.snake or
                                                     ((new_snake_head[0],
                                                       new_snake_head[1] - 1) not in self.empty_cells and
                                                      (new_snake_head[0], new_snake_head[1] - 1) != self.dot)))
                 or (self.snake_head_direc == 2 and ((new_snake_head[0] - 1, new_snake_head[1]) in self.snake or
                                                     ((new_snake_head[0] - 1,
                                                       new_snake_head[1]) not in self.empty_cells and
                                                      (new_snake_head[0] - 1, new_snake_head[1]) != self.dot)))
                 or (self.snake_head_direc == 3 and ((new_snake_head[0], new_snake_head[1] + 1) in self.snake or
                                                     ((new_snake_head[0],
                                                       new_snake_head[1] + 1) not in self.empty_cells and
                                                      (new_snake_head[0], new_snake_head[1] + 1) != self.dot)))),
                # danger right of snake head
                self.snake_head_direc == 3,  # snake direction left
                self.snake_head_direc == 1,  # snake direction right
                self.snake_head_direc == 0,  # snake direction up
                self.snake_head_direc == 2,  # snake direction down
                (new_snake_head[0] > self.dot[0]),  # food left
                (new_snake_head[0] < self.dot[0]),  # food right
                (new_snake_head[1] < self.dot[1]),  # food up
                (new_snake_head[1] > self.dot[1]),  # food down

            ]

            for i in range(len(self.state)):
                if self.state[i]:
                    self.state[i] = 1
                else:
                    self.state[i] = 0

            # TODO CHANGED
            # self.state = np.zeros((self.height, self.width))
            # self.state = np.pad(self.state[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), mode='constant',
            #                 constant_values=-1)
            #
            # for x, y in self.snake:
            #     self.state[y, x] = 1
            #
            # self.state[self.dot[1], self.dot[0]] = 2

        return [np.asarray(self.state), self.reward, self.done, self.add]

    def reset(self):
        self.done = 0
        self.score = 0
        self.reward = 0
        self.snake_head_direc = 0

        self.snake = copy.deepcopy(self.initial_snake)

        self.empty_cells = {(x, y) for x in range(self.width) for y in range(self.height)}
        for i in self.snake:
            self.empty_cells.remove(i)

        self.dot = random.choice(tuple(self.empty_cells))
        self.empty_cells.remove(self.dot)

        # TODO CHANGED
        # self.state = np.zeros((self.height, self.width))
        # self.state = np.pad(self.state[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), mode='constant',
        #                     constant_values=-1)
        # for x, y in self.snake:
        #     self.state[y, x] = 1
        #
        # self.state[self.dot[1], self.dot[0]] = 2

        snake_head = self.snake.pop()

        self.state = [
            False,  # danger left
            False,  # danger straight
            False,  # danger right
            # TODO changed! start with snake with 1 point
            # (self.snake_head_direc == 3),  # snake direction left
            # (self.snake_head_direc == 1),  # snake direction right
            # (self.snake_head_direc == 0),  # snake direction up
            # (self.snake_head_direc == 2),  # snake direction down
            False,
            False,
            True,
            False,
            (snake_head[0] > self.dot[0]),  # food left
            (snake_head[0] < self.dot[0]),  # food right
            (snake_head[1] < self.dot[1]),  # food up
            (snake_head[1] > self.dot[1]),  # food down

        ]

        for i in range(len(self.state)):
            if self.state[i]:
                self.state[i] = 1
            else:
                self.state[i] = 0

        self.snake.append(snake_head)

        return np.asarray(self.state)

    def render(self, mode='human', close=False):
        width = 600
        height = width
        width_scaling_fact = width / self.width
        height_scaling_fact = height / self.height

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        # give snake head a different color
        snake_head = self.snake.pop()
        x, y = snake_head
        l, r, t, b = x * width_scaling_fact, (x + 1) * width_scaling_fact, y * height_scaling_fact, \
                     (y + 1) * height_scaling_fact
        square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        square.set_color(0, 0, 205)
        self.viewer.add_onetime(square)

        if len(self.snake) > 0:
            for x, y in self.snake:
                l, r, t, b = x * width_scaling_fact, (x + 1) * width_scaling_fact, y * height_scaling_fact, (
                        y + 1) * height_scaling_fact
                square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                square.set_color(0, 0, 0)
                self.viewer.add_onetime(square)

            #   append snake head to snake again after snake body has been colored black
        self.snake.append(snake_head)

        if self.dot:
            x, y = self.dot
            l, r, t, b = x * width_scaling_fact, (x + 1) * width_scaling_fact, y * height_scaling_fact, (
                    y + 1) * height_scaling_fact
            square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            square.set_color(1, 0, 0)
            self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
