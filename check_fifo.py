from francis import *

a = tf.random_normal((4,))
b = tf.random_normal((1,))

fifo = tf.FIFOQueue(10, (tf.float32, tf.float32), ((4,), (1,)))
enqueue_op = fifo.enqueue((a, b))
dequeued = fifo.dequeue_many(10)

with tf.Session() as session:
    for i in range(10):
        session.run(enqueue_op)
    print([a.shape for a in session.run(dequeued)])
