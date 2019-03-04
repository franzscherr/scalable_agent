from francis import *
import py_process
import gym
import threading
from tensorflow.python.util import nest


class GymEnvironment(object):
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


def build_actor(env):
    obs, state = env.initial()

    def _step(_previous, _elems):
        return env.step(_elems, _previous[-1])

    result = tf.scan(_step, tf.zeros((1000,), tf.int32), initializer=(tf.zeros(()), tf.zeros((), tf.bool), obs, state))
    return result


class QueueHook(tf.train.SessionRunHook):
    def __init__(self):
        super().__init__()
        self.threads = []
        self.should_stop = False

    def after_create_session(self, session, coord):
        tf.logging.info('Starting queuerunners')

        def _enqueue(_op):
            while not self.should_stop:
                try:
                    session.run(_op)
                except:
                    pass

        for enqueue_op in tf.get_collection('enqueue_ops'):
            thread = threading.Thread(target=_enqueue, args=(enqueue_op,))
            self.threads.append(thread)
            thread.start()

    def end(self, session):
        tf.logging.info('Stopping all queuerunners')
        self.should_stop = True
        for thread in self.threads:
            thread.join()


if __name__ == '__main__':
    n_actors = 16
    n_batch = 10

    p = py_process.PyProcess(GymEnvironment, 'CartPole-v0', observation_shape=(4,))
    env = GymFlow(p.proxy)
    structure = build_actor(env)
    flattened_structure = nest.flatten(structure)
    dtypes = [a.dtype for a in flattened_structure]
    shapes = [a.shape.as_list() for a in flattened_structure]
    queue = tf.FIFOQueue(1, dtypes, shapes)

    for j in range(n_actors):
        p = py_process.PyProcess(GymEnvironment, 'CartPole-v0', observation_shape=(4,))
        env = GymFlow(p.proxy)
        actor_output = build_actor(env)
        enqueue_op = queue.enqueue(nest.flatten(actor_output))
        tf.add_to_collection('enqueue_ops', enqueue_op)

    dequeued = queue.dequeue_many(n_batch)
    queue_hook = QueueHook()
    process_hook = py_process.PyProcessHook()

    with tf.train.MonitoredSession(hooks=[
        process_hook,
        queue_hook
    ]) as session:
        for i in range(100):
            session.run(dequeued)
