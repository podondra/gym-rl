import numpy
import tensorflow
from collections import deque


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
        reshape_img = tensorflow.reshape(resize_img, [84, 84])
        # TODO divede by 255?
        self.preprocessor = reshape_img

    def __call__(self, S, sess):
        return sess.run(self.preprocessor, {self.img: S})


class ReplayMemory:
    # there is size of 1M in the Nature's paper
    REPLAY_MEMORY_SIZE = 1000000
    REPLAY_START_SIZE = 50000
    AGENT_HISTORY_LEN = 4
    MINIBATCH_SIZE = 32

    def __init__(self, env, preprocess, sess):
        """Initialize replay memory.

        Initialize deque for Python's collections to REPLAY_MEMORY_SIZE size
        and populate it with random experience.

        env is OpenAI gym environment,
        preprocess is instance of Preprocessor class,
        sess is TensorFlow session.
        """
        self.D = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.populate_memory(env, preprocess, sess)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, index):
        return self.D[index]

    def populate_memory(self, env, preprocess, sess):
        """Populate replay memory with random experience."""
        S = env.reset()
        theta = preprocess(S, sess)
        while len(self.D) < self.REPLAY_START_SIZE:
            A = env.action_space.sample()
            S_next, R, done, _ = env.step(A)
            theta_next = preprocess(S_next, sess)
            self.store(theta, A, R)
            S, theta = S_next, theta_next
            if done and len(self.D) < self.REPLAY_START_SIZE:
                # store terminal state with action as -1 and zero reward
                self.store(theta, -1, 0)
                S = env.reset()
                theta = preprocess(S, sess)

    def store(self, theta, A, R):
        """Store transition in replay memory."""
        self.D.append((theta, A, R))

    def get_state(self, index):
        return numpy.stack(
                [self.D[i][0] for i in range(index - 3, index + 1)],
                axis=-1
                )

    def get_recent_state(self):
        return self.get_state(len(self.D) - 1)

    def get_transition(self, index):
        return (
                self.get_state(index - 1),
                self.D[index][1],
                self.D[index][2],
                self.get_state(index)
                )

    def sample_minibatch(self):
        """Sample random minibatch of size 32 from replay memory."""
        indexes = numpy.random.randint(
                low=self.AGENT_HISTORY_LEN, high=len(self.D),
                size=self.MINIBATCH_SIZE
                )
        # TODO handle terminal states correctly
        # for example there should not be terminal state
        # in the AGENT_HISTORY_LEN frames
        # TODO numpy.stack for demands of gradient computation
        return [self.get_transition(index) for index in indexes]


class DQN:
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
