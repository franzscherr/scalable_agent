from francis import *
import py_process
import gym
from tensorflow.python.util import nest


class GymEnvironment():
    def __init__(self, name, **kwargs):
        self.env = gym.make(name)

    def initial(self):
        observation = self.env.reset()
        return np.float32(observation)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        if done:
            observation = self.env.reset()
        reward = np.array(reward, dtype=np.float32)
        return reward, done, np.float32(observation)

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of `TensorSpec` with the method's output specification."""
        observation_spec = tf.TensorSpec(constructor_kwargs['observation_shape'], tf.float32)

        if method_name == 'initial':
            return observation_spec
        elif method_name == 'step':
            return (
                tf.TensorSpec([], tf.float32),
                tf.TensorSpec([], tf.bool),
                observation_spec,
            )


class GymFlow(object):
    def __init__(self, env):
        self.env = env

    def initial(self):
        observation = self.env.initial()
        with tf.control_dependencies([observation]):
            state = tf.zeros(())
        return observation, state

    def step(self, action, state):
        with tf.control_dependencies([state]):
            reward, done, observation = self.env.step(action)
            state = tf.zeros(()) * reward
        return reward, done, observation, state


if __name__ == '__main__':
    results = []
    ps = []
    enqueue_ops = []
    for j in range(16):
        p = py_process.PyProcess(GymEnvironment, 'CartPole-v0', observation_shape=(4,))
        ps.append(p)
        env = GymFlow(p.proxy)
        p.start()
        obs, state = env.initial()

        def _step(_previous, _elems):
            return env.step(_elems, _previous[-1])

        result = tf.scan(_step, tf.zeros((1000,), tf.int32), initializer=(tf.zeros(()), tf.zeros((), tf.bool), obs, state))
        dtypes = [a.dtype for a in nest.flatten(result)]
        shapes = [a.shape.as_list() for a in nest.flatten(result)]
        queue = tf.FIFOQueue(2, dtypes, shapes)
        enqueue_op = queue.enqueue(nest.flatten(result))
        enqueue_ops.append(enqueue_op)
    print(enqueue_ops)
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    dequeued = queue.dequeue_many(32)
    # rew, d, obs = env.step(tf.zeros((), tf.int32))
    with tf.Session() as session:
        tf.train.start_queue_runners(session)
        print(session.run(dequeued))
        [p.close(session) for p in ps]
