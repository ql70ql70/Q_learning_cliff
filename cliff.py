# import numpy
# from gym import spaces
# import gym

# class Cliff(gym.Env):


class Cliff:
    x = -999.0  # 悬崖
    y = -99.0  # 有坑,费时
    S = -9.0  # 出发地
    w = -29.0  # 小坑
    o = -1.0  # 大路
    a = 9999.0  # 终点1,低能
    A = 19999.0  # 终点2,高能

    #     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15
    l0 = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]  # 0
    l1 = [x, a, x, o, o, o, x, y, o, o, o, o, o, o, o, x]  # 1
    l2 = [x, o, x, w, o, o, y, o, o, o, o, o, o, o, o, x]  # 2
    l3 = [x, y, o, y, x, x, o, o, x, x, x, x, x, o, o, x]  # 3
    l4 = [x, o, o, o, y, x, x, o, x, o, o, o, o, o, o, x]  # 4
    l5 = [x, y, o, o, y, o, y, o, x, y, o, o, x, x, x, x]  # 5
    l6 = [x, y, x, x, x, x, x, o, x, x, o, o, o, o, o, x]  # 6
    l7 = [x, o, o, x, x, x, x, o, x, x, x, o, o, o, o, x]  # 7
    l8 = [x, o, o, o, o, o, x, o, o, o, x, x, x, x, o, x]  # 8
    l9 = [x, x, o, o, x, x, x, x, x, o, o, o, x, o, o, x]  # 9
    la = [x, x, o, o, x, x, x, x, x, x, o, o, x, o, y, x]  # 10
    lb = [x, o, o, o, o, o, y, x, o, x, o, o, x, x, A, x]  # 11
    lc = [x, o, o, x, x, x, y, y, o, o, o, o, x, x, x, x]  # 12
    ld = [x, o, o, x, x, x, y, y, o, a, x, x, x, o, x, x]  # 13
    le = [x, o, o, x, o, x, x, x, x, x, x, x, y, o, S, x]  # 14
    lf = [x, o, o, x, o, o, o, o, o, o, x, o, y, x, x, x]  # 15
    lg = [x, o, o, x, x, o, x, o, o, x, x, x, o, o, o, x]  # 16
    lh = [x, o, y, x, x, o, x, x, o, x, o, x, x, y, y, x]  # 17
    li = [x, o, o, o, o, o, x, x, o, o, o, o, o, o, o, x]  # 18
    lj = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]  # 19

    rewards = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9,
               la, lb, lc, ld, le, lf, lg, lh, li, lj]

    def __init__(self):
        self.states = []
        self.states_actions_table = {(0, 0): ['70']}
        self.len_x = len(Cliff.l0)
        self.len_y = len(Cliff.rewards)
        for i in range(0, self.len_y):
            for j in range(0, self.len_x):
                if Cliff.rewards[i][j] != Cliff.x:
                    self.states.append((i, j))
                    self.states_actions_table[(i, j)] = []

                    if Cliff.rewards[i + 1][j] != Cliff.x:
                        self.states_actions_table[(i, j)].append('s')
                    if Cliff.rewards[i][j + 1] != Cliff.x:
                        self.states_actions_table[(i, j)].append('e')
                    if Cliff.rewards[i - 1][j] != Cliff.x:
                        self.states_actions_table[(i, j)].append('n')
                    if Cliff.rewards[i][j - 1] != Cliff.x:
                        self.states_actions_table[(i, j)].append('w')
                    """"""
                    if Cliff.rewards[i - 1][j + 1] != Cliff.x:
                        self.states_actions_table[(i, j)].append('ne')
                    if Cliff.rewards[i - 1][j - 1] != Cliff.x:
                        self.states_actions_table[(i, j)].append('nw')
                    if Cliff.rewards[i + 1][j + 1] != Cliff.x:
                        self.states_actions_table[(i, j)].append('se')
                    if Cliff.rewards[i + 1][j - 1] != Cliff.x:
                        self.states_actions_table[(i, j)].append('sw')

                if Cliff.rewards[i][j] == Cliff.S:
                    self.origin = [i, j]

        self.delta = {'n': [-1, 0], 's': [1, 0], 'e': [0, 1], 'w': [0, -1],
                      'ne': [-1, 1], 'nw': [-1, -1], 'se': [1, 1], 'sw': [1, -1]}

        self.gamma = 1
        self.state = [0, 0]
        self.next_state = [0, 0]

    def get_gamma(self):
        return self.gamma

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.states_actions_table

    def get_size_of_map(self):
        return self.len_y, self.len_x

    def step(self, action):
        key = "%d_%d_%s" % (self.state[0], self.state[1], action)
        now_delta = self.delta[action]
        self.next_state[0] = now_delta[0] + self.state[0]
        self.next_state[1] = now_delta[1] + self.state[1]

        reward = Cliff.rewards[self.next_state[0]][self.next_state[1]]

        if now_delta[0] * now_delta[1] != 0:
            reward -= 0.4142

        self.state = self.next_state
        return key, self.next_state, reward

    def reset(self):
        self.state[0] = self.origin[0]
        self.state[1] = self.origin[1]
        return self.state
