import numpy
import tensorflow


class Preprocessor:
    def __init__(self):
        self.img = tensorflow.placeholder(
                tensorflow.uint8,
                shape=[210, 160, 3]
                )
        gray_img = tensorflow.image.rgb_to_grayscale(self.img)
        resize_img = tensorflow.image.resize_images(
                gray_img,
                [84, 84],
                method=tensorflow.image.ResizeMethod.BILINEAR
                )
        self.preprocessor = resize_img

    def __call__(self, S, sess):
        return sess.run(self.preprocessor, {self.img: S})


class ReplayMemory:
    REPLAY_MEMORY_SIZE = 1000000
    REPLAY_START_SIZE = 50000
    AGENT_HISTORY_LEN = 4

    def __init__(self):
        ...


class DQN:
    MINIBATCH_SIZE = 32
    GAMMA = 0.99  # discount factor
    ALHPA = 0.00025  # learning rate
    GRADIENT_MOMENTUM = 0.95
    MIN_SQRT_MOMENTUM = 0.01
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
        # TODO dtype and divide by 255
        X = tensorflow.placeholder(shape=[self.MINIBATCH_SIZE, 84, 84, 4])
        y = tensorflow.placeholder(tensorflow.float, shape=[MINIBATCH_SIZE])

        conv_1 = tensorflow.layers.conv2d(
                inputs=X, filters=32, kernel_size=8, strides=4,
                activation=tensorflow.nn.relu
                )
        conv_2 = tensorflow.layers.conv2d(
                inputs=conv_1, filters=64, kernel_size=4, strides=2,
                activation=tensorflow.nn.relu
                )
        conv_3 = tensorflow.layers.conv2d(
                inputs=conv_2, filters=64, kernel_size=3,
                activation=tensorflow.nn.relu
                )
        flat = tensorflow.layers.flatten(conv_3)
        fc = tensorflow.layers.dense(
                flat, units=512, activation=tensorflow.nn.relu
                )
        output = tensorflow.layers.dense(fc, units=self.env.action_space.n)

        loss = tensorflow.reduce_mean(
            tensorflow.squared_difference(y, )
        )

        optimizer = tensorflow.train.RMSPropOptimizer(
            learning_rate=self.ALHPA,
            decay=1,  # do not decay learning rate
            momentum=self.GRADIENT_MOMENTUM,
            epsilon=self.MIN_SQRT_MOMENTUM,
        )

    def clone_weights(self):
        ...

    def get_q_update(self):
        ...

    def gradient_descent_step(self):
        ...

    def epsilon_greedy_action(self, S):
        epsilon = next(self.epsilons)
        # TODO repeat action 4 times
        if numpy.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            # TODO a_t = \argmax_a Q(\theta(s_t), a; \Theta)
            return 0
