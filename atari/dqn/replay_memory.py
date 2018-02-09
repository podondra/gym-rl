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
