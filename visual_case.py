import tensorflow as tf
import numpy as np
import gym
import time

# 定义超参数
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # actor学习率
LR_C = 0.002  # critic学习率
GAMMA = 0.9  # 累计折扣奖励因子
TAU = 0.01  # 软更新tao
MEMORY_CAPACITY = 10000  # buffer R, 经验回放容器
BATCH_SIZE = 32  # 每批随机读取批次大小

RENDER = False
ENV_NAME = 'Pendulum-v0'


# 定义DDPG类
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        # memory 存放的是序列（s,a,r,s+1）= s*2+a+1(r=1)
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # 建立网络，actor网络输入是S,critic输入是s,a
        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        # soft updating
        """
        tf.train.ExponentialMovingAverage(decay)是采用滑动平均的方法更新参数。这个函数初始化需要提供一个衰减速率（decay），用于控制模型的更新速度。这个函数还会维护一个影子变量（也就是更新参数后的参数值），这个影子变量的初始值就是这个变量的初始值，影子变量值的更新方式如下：
   shadow_variable = decay * shadow_variable + (1-decay) * variable
   shadow_variable是影子变量，variable表示待更新的变量，也就是变量被赋予的值，decay为衰减速率。decay一般设为接近于1的数（0.99,0.999）。decay越大模型越稳定，因为decay越大，参数更新的速度就越慢，趋于稳定。
        """

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    # 选取动作函数
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    # 从R buffer中学习
    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    # 存储序列
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # 建立actor网络（输入S_dim，输出a_dim, 采用tanh激活函数）
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    # 建立critic网络（输入S_dim，s_dim, 输出q）
    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


# training process

# 环境初始化
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

# 获取s,a的维度
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # 定义探索因子

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # 添加探索噪音
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)  # 随机选取动作探索
        # np.clip()函数是，如果随机生成的数字大于2，则为2 ，如果小于-2，则为-2，其他则为本身

        s_, r, done, info = env.step(a)
        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995  # 减缓动作探索度，即衰减速率
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - t1)