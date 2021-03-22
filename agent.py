from cliff import Cliff
import random
import copy
import os


class Agent:
    cliff_70 = Cliff()
    cliff_70.reset()

    gamma = cliff_70.get_gamma()
    states = cliff_70.get_states()
    states_actions_table = cliff_70.get_actions()
    a = Cliff.a

    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

        runnable = '1'

        while runnable == '1':
            to_do = input("If you want to let me begin learning, please give me 1.\n" +
                          "Or, I'll show you the map I've gotten.\n")

            q_func_70 = {}

            if to_do == '1':
                do_refresh = input("If you want to refresh the q-table, please give me 1.\n")
                do_double = input("If you want to do double Q-learning, please give me 1.\n")

                for state in Agent.states:
                    actions = Agent.states_actions_table[state]
                    for action in actions:
                        key = "%d_%d_%s" % (state[0], state[1], action)
                        q_func_70[key] = 0.0

                if do_double == '1':
                    if do_refresh == '1':
                        f_write_l = open("best_q_func_l", mode='w')
                        f_write_r = open("best_q_func_r", mode='w')
                        self.write_q_tables(f_write_l, f_write_r, q_func_70, q_func_70)
                        f_write_l.close()
                        f_write_r.close()

                    f_read_l = open("best_q_func_l", mode='r')
                    f_read_r = open("best_q_func_r", mode='r')
                    q_func_70_l = self.read_q_table(f_read_l)
                    q_func_70_r = self.read_q_table(f_read_r)
                    f_read_l.close()
                    f_read_r.close()

                    for state in Agent.states:
                        actions = Agent.states_actions_table[state]
                        for action in actions:
                            key = "%d_%d_%s" % (state[0], state[1], action)
                            q_func_70[key] = (q_func_70_l[key] + q_func_70_r[key]) / 2

                    q_func_70 = self.double_q_learning_episodes(q_func_70, alpha, epsilon, q_func_70_l, q_func_70_r)
                    """"""
                else:
                    if do_refresh == '1':
                        f_write = open("best_q_func", mode='w')
                        self.write_q_table(f_write, q_func_70)
                        f_write.close()

                    f_read = open("best_q_func", mode='r')
                    q_func_70 = self.read_q_table(f_read)
                    f_read.close()

                    q_func_70 = self.q_learning_episodes(q_func_70, alpha, epsilon)
                    """"""

            else:
                show_map = input("If you want to watch the double Q map, please give me 1.\n")
                if show_map == '1':
                    if os.path.getsize("best_q_func_0") == 0:
                        print("I do not have a double Q map.")
                    else:
                        f_read = open("best_q_func_0", mode='r')
                        q_func_70 = self.read_q_table(f_read)
                        f_read.close()

                else:
                    if os.path.getsize("best_q_func") == 0:
                        print("I do not have a Q map.")
                    else:
                        f_read = open("best_q_func", mode='r')
                        q_func_70 = self.read_q_table(f_read)
                        f_read.close()

            path_70, demo_map_70, reward_sum_70 = self.q_func_demo(q_func_70)
            print(path_70)
            print('')
            self.draw_map(demo_map_70)
            print(reward_sum_70)

            runnable = input("If you have something else need me to do, please give me 1.\n" +
                             "Or, I'll close the program.\n")

    # 读取已有的q表
    @staticmethod
    def read_q_table(f):
        best_q_func = {}

        for f_line in f:
            if len(f_line) == 0:
                continue
            else:
                best_q = f_line.split(":")
                best_q_func[best_q[0]] = float(best_q[1])
        return best_q_func

    # 写一个q表
    @staticmethod
    def write_q_table(f, q_func):
        for key_i in q_func:
            f.write(key_i + ":" + str(q_func[key_i]) + "\n")

    # 写两个q表
    def write_q_tables(self, f_l, f_r, q_func_l, q_func_r):
        self.write_q_table(f_l, q_func_l)
        self.write_q_table(f_r, q_func_r)

    # 画地图
    @staticmethod
    def draw_map(demo_map):
        for line in demo_map:
            str_line = ""
            for m in line:
                str_line += m + "  "
            print(str_line)

    # 获得某一q表在环境下的最佳轨迹及其奖励值
    def q_func_demo(self, q_func):
        path = []
        now_state = Agent.cliff_70.reset()
        reward = 0
        reward_sum = 0
        path.append([now_state[0], now_state[1]])
        print(now_state)

        demo_y, demo_x = Agent.cliff_70.get_size_of_map()
        demo_map = []
        for i in range(0, demo_y):
            demo_line = []
            for j in range(0, demo_x):
                demo_line.append(' ')
                if Cliff.rewards[i][j] == Cliff.x:
                    demo_line[j] = 'x'
            demo_map.append(demo_line)

        demo_map[now_state[0]][now_state[1]] = '0'

        while reward < Agent.a - 1:
            now_action = self.greedy(q_func, now_state)  # greedy选择当前动作action
            now_key, next_state, reward = Agent.cliff_70.step(
                now_action)  # 对环境做出当前动作action,得到当前的key,下一个状态next_state,和回报reward
            reward_sum += reward
            now_state = next_state  # 向后走一步
            path.append([now_state[0], now_state[1]])
            demo_map[now_state[0]][now_state[1]] = 'o'
            print(now_state)

        demo_map[now_state[0]][now_state[1]] = 'A'

        return path, demo_map, reward_sum

    # 计算当前q表与基准q表的误差
    @staticmethod
    def compute_error(q_func, q_func_0):
        sum_delta = 0.0
        # print(q_func)
        # print(q_func_0)
        for q_func_key in q_func:
            error = q_func[q_func_key] - q_func_0[q_func_key]

            sum_delta += error * error
        return sum_delta

    # 计算当前q表最佳路径的总奖励与基准q表最佳路径的总奖励的误差
    def compute_reward_error(self, q_func, q_func_0):
        path_, map_, reward_ = self.q_func_demo(q_func)
        path_0, map_0, reward_0 = self.q_func_demo(q_func_0)
        return reward_ - reward_0

    # 贪婪策略选择某一状态和某一Q表下的动作
    @staticmethod
    def greedy(q_func, now_state):
        act_max_i = []
        a = now_state[0]
        b = now_state[1]
        now_actions = Agent.states_actions_table[(a, b)]
        key_i = "%d_%d_%s" % (a, b, now_actions[0])
        q_max = q_func[key_i]
        len_act = len(now_actions)

        for i in range(1, len_act):  # 扫描动作空间得到最大动作值
            key_i = "%d_%d_%s" % (a, b, now_actions[i])
            if q_max < q_func[key_i]:
                q_max = q_func[key_i]

        for i in range(0, len_act):  # 扫描动作空间得到最大动作值对应的动作
            key_i = "%d_%d_%s" % (a, b, now_actions[i])
            if q_func[key_i] == q_max:
                act_max_i.append(i)
        # 返回最大动作值对应的动作中的任意一个
        return now_actions[act_max_i[random.randint(0, len(act_max_i) - 1)]]

    # epsilon贪婪策略选择某一状态和某一Q表下的动作
    @staticmethod
    def epsilon_greedy(q_func, now_state, epsilon):
        act_max_i = []
        a = now_state[0]
        b = now_state[1]
        now_actions = Agent.states_actions_table[(a, b)]
        key_i = "%d_%d_%s" % (a, b, now_actions[0])
        q_max = q_func[key_i]
        len_act = len(now_actions)

        for i in range(1, len_act):  # 扫描动作空间得到最大动作值
            key_i = "%d_%d_%s" % (a, b, now_actions[i])
            if q_max < q_func[key_i]:
                q_max = q_func[key_i]

        for i in range(0, len_act):  # 扫描动作空间得到最大动作值对应的动作
            key_i = "%d_%d_%s" % (a, b, now_actions[i])
            if q_func[key_i] == q_max:
                act_max_i.append(i)

        select = random.random()

        # 模式选择信号小于epsilon,就在全部动作里面选一个
        if select < epsilon:
            return now_actions[random.randint(0, len_act - 1)]
        # 否则,在最优动作里选一个
        else:
            return now_actions[act_max_i[random.randint(0, len(act_max_i) - 1)]]

    # 智能体在环境里跑一个episode,得到新的q表
    def q_learning_episode(self, q_func, alpha, epsilon):
        now_state = Agent.cliff_70.reset()
        reward = 0

        while reward < Agent.a - 1:
            now_action = self.epsilon_greedy(q_func, now_state, epsilon)  # epsilon_greedy选择当前动作action
            now_key, next_state, reward = Agent.cliff_70.step(
                now_action)  # 对环境做出当前动作action,得到当前的key,下一个状态next_state,和回报reward
            # print(now_state, action, reward)
            # inf = input("")

            if reward > Agent.a - 1:
                q_func[now_key] = q_func[now_key] + alpha * (reward + Agent.gamma * reward - q_func[now_key])
            else:
                action_expect = self.greedy(q_func, next_state)  # 在下一个状态next_state有由贪婪策略得到的预期动作action_expect
                key_expect = "%d_%d_%s" % (next_state[0], next_state[1], action_expect)  # 生成预期动作的key_expect

                delta_q_func = reward + Agent.gamma * q_func[key_expect] - q_func[now_key]
                q_func[now_key] = q_func[now_key] + alpha * delta_q_func
                # 更新当前状态下,当前选择动作的q表
                now_state = next_state  # 向后走一步

        return q_func

    # double-Q-learning算法下,智能体在环境里跑一个episode,得到新的q表
    def double_q_learning_episode(self, q_func, alpha, epsilon, q_func_l, q_func_r):
        now_state = Agent.cliff_70.reset()
        reward = 0

        while reward < Agent.a - 1:
            now_action = self.epsilon_greedy(q_func, now_state, epsilon)  # epsilon_greedy选择当前动作action
            now_key, next_state, reward = Agent.cliff_70.step(
                now_action)  # 对环境做出当前动作action,得到当前的key,下一个状态next_state,和回报reward
            # print(now_state, action, reward)
            # inf = input("")

            sel = random.randint(0, 1)

            if sel == 0:
                q_func_master = q_func_l
                q_func_slave = q_func_r
            else:
                q_func_master = q_func_r
                q_func_slave = q_func_l

            if reward > Agent.a - 1:
                q_func_master[now_key] = q_func_master[now_key] + alpha * (
                        reward + Agent.gamma * reward - q_func_master[now_key])
            else:
                action_expect = self.greedy(q_func_master, next_state)  # 在下一个状态next_state有由贪婪策略得到的预期动作action_expect
                key_expect = "%d_%d_%s" % (next_state[0], next_state[1], action_expect)  # 生成预期动作的key_expect

                q_func_master[now_key] = q_func_master[now_key] + alpha * (
                        reward + Agent.gamma * q_func_slave[key_expect] - q_func_master[now_key])
                # 更新当前状态下,当前选择动作的q表

                now_state = next_state  # 向后走一步

            q_func[now_key] = (q_func_l[now_key] + q_func_r[now_key]) / 2

        return q_func, q_func_l, q_func_r

    # 智能体跑多个episode,迭代更新q表
    def q_learning_episodes(self, q_func_0, alpha, epsilon):
        i = 0
        delta = []
        for k in range(0, 100):
            delta.append(1)
        sum_delta = 100.0

        # for j in range(0, 10000):
        while sum_delta != 0.0:
            i += 1
            # print(q_func_0)
            q_func = copy.deepcopy(q_func_0)
            q_func = self.q_learning_episode(q_func, alpha, epsilon)
            # print(q_func)
            now_delta = self.compute_error(q_func, q_func_0)
            q_func_0 = q_func

            delta.append(now_delta)
            del delta[0]

            sum_delta = 0.0
            for k in range(0, 100):
                sum_delta += delta[k]

            print(i, now_delta, sum_delta)

        f_write = open("best_q_func", mode='w')
        self.write_q_table(f_write, q_func_0)
        f_write.close()

        return q_func_0

    # double-Q-learning算法下,智能体跑多个episode,迭代更新q表
    def double_q_learning_episodes(self, q_func_0, alpha, epsilon, q_func_l_0, q_func_r_0):
        i = 0
        delta = []
        for k in range(0, 100):
            delta.append(1)
        sum_delta = 100.0

        for j in range(0, 10000):
        # while sum_delta != 0.0:
            i += 1
            # print(q_func_0)
            q_func = copy.deepcopy(q_func_0)
            q_func, q_func_l_0, q_func_r_0 = \
                self.double_q_learning_episode(q_func, alpha, epsilon, q_func_l_0, q_func_r_0)
            # print(q_func)
            now_delta = self.compute_error(q_func, q_func_0)
            q_func_0 = q_func

            delta.append(now_delta)
            del delta[0]

            sum_delta = 0.0
            for k in range(0, 100):
                sum_delta += delta[k]

            print(i, now_delta, sum_delta)

        f_write_l = open("best_q_func_l", mode='w')
        f_write_r = open("best_q_func_r", mode='w')
        self.write_q_tables(f_write_l, f_write_r, q_func_l_0, q_func_r_0)
        f_write_l.close()
        f_write_r.close()

        f_write = open("best_q_func_0", mode='w')
        self.write_q_table(f_write, q_func_0)
        f_write.close()

        return q_func_0
