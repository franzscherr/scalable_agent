from francis import *
import dynamic_batching


@dynamic_batching.batch_fn_with_options(
    minimum_batch_size=2, timeout_ms=100)
def fn(a, b):
    return a + b


output0 = fn(tf.constant([1]), tf.constant([2]))  # Will be batched with the
# next call.
output1 = fn(tf.constant([3]), tf.constant([4]))
output2 = fn(tf.constant([3]), tf.constant([4]))

with tf.Session() as session:
    tf.train.start_queue_runners()
    print(session.run([output0, output1, output2]))
