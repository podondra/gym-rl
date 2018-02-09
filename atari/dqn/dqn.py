import numpy
import tensorflow


class DQN:
    GAMMA = 0.99  # discount factor
    ALHPA = 0.00025  # rmsprop learning rate
    GRADIENT_MOMENTUM = 0.95  # rmsprop parameter
    MIN_SQRT_MOMENTUM = 0.01  # rmsprop parameter
    INITIAL_EPSILON = 1
    FINAL_EPSILON = 0.1
    FINAL_EXPLORATION_FRAME = 1000000
    ACTION_REPEAT = 4
    UPDATE_FREQUENCY = 4

    def __init__(self, env, weights=None):
        self.env = env
        self.dqn = self.create_dqn()
        self.weights = None  # TODO
        self.epsilons = iter(numpy.linspace(
            self.INITIAL_EPSILON, self.FINAL_EPSILON,
            num=self.FINAL_EXPLORATION_FRAME
            ))

    def create_dqn(self):
        self.Xs = tensorflow.placeholder(
                tensorflow.uint8, shape=[None, 84, 84, 4]
                )
        float_Xs = tensorflow.to_float(self.Xs) / 255
        conv_1 = tensorflow.layers.conv2d(
                float_Xs, filters=32, kernel_size=8, strides=4,
                activation=tensorflow.nn.relu
                )
        conv_2 = tensorflow.layers.conv2d(
                conv_1, filters=64, kernel_size=4, strides=2,
                activation=tensorflow.nn.relu
                )
        conv_3 = tensorflow.layers.conv2d(
                conv_2, filters=64, kernel_size=3,
                activation=tensorflow.nn.relu
                )
        flat = tensorflow.layers.flatten(conv_3)
        fc = tensorflow.layers.dense(
                flat, units=512, activation=tensorflow.nn.relu
                )
        q_values = tensorflow.layers.dense(
                fc, units=self.env.action_space.n
                )

        self.argmax_action = tensorflow.argmax(q_values, axis=1)
        self.max_action = tensorflow.max(q_values, axis=1)

        self.ys = tensorflow.placeholder(
                tensorflow.float, shape=[self.MINIBATCH_SIZE]
                )
        self.actions = tensorflow.placeholder(
                tensorflow.int, shape=[self.MINIBATCH_SIZE]
                )
        self.loss = tensorflow.reduce_mean(
                tensorflow.squared_difference(
                    self.ys,
                    tensorflow.gather(q_values, self.actions, axis=1)
                    )
                )
        optimizer = tensorflow.train.RMSPropOptimizer(
            learning_rate=self.ALHPA,
            decay=1,  # do not decay learning rate
            momentum=self.GRADIENT_MOMENTUM,
            epsilon=self.MIN_SQRT_MOMENTUM,
        )
        # gradient operator
        self.train = optimizer.minimize(self.loss)

    def clone_weights(self):
        ...

    def max_q_values(self, Ss, sess):
        return sess.run(self.max_action, {self.Xs: Ss})

    def gradient_descent_step(self, ys, Ss, As, sess):
        sess.run(self.train, {self.ys: ys, self.Xs: Ss, self.actions: As})

    def epsilon_greedy_action(self, S, sess):
        epsilon = next(self.epsilons)
        # TODO repeat action 4 times
        if numpy.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            # a_t = \argmax_a Q(\theta(s_t), a; \Theta)
            return sess.run(self.argmax_action, {self.X: S})
