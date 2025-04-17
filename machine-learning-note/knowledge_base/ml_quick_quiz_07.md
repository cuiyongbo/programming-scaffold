```
>>> import tensorflow as tf
>>> sess = tf.Session()
>>> help(sess.run)

Help on method run in module tensorflow.python.client.session:

run(self, fetches, feed_dict=None, options=None, run_metadata=None) method of tensorflow.python.client.session.Session instance
    Runs operations and evaluates tensors in `fetches`.
    
    This method runs one "step" of TensorFlow computation, by
    running the necessary graph fragment to execute every `Operation`
    and evaluate every `Tensor` in `fetches`, substituting the values in
    `feed_dict` for the corresponding input values.
    
    The `fetches` argument may be a single graph element, or an arbitrarily
    nested list, tuple, namedtuple, dict, or OrderedDict containing graph
    elements at its leaves.  A graph element can be one of the following types:
    
    * A `tf.Operation`.
      The corresponding fetched value will be `None`.
    * A `tf.Tensor`.
      The corresponding fetched value will be a numpy ndarray containing the
      value of that tensor.
    * A `tf.SparseTensor`.
      The corresponding fetched value will be a
      `tf.compat.v1.SparseTensorValue`
      containing the value of that sparse tensor.
    * A `get_tensor_handle` op.  The corresponding fetched value will be a
      numpy ndarray containing the handle of that tensor.
    * A `string` which is the name of a tensor or operation in the graph.
    
    The value returned by `run()` has the same shape as the `fetches` argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.
    
    Example:
    
    ```python
       a = tf.constant([10, 20])
       b = tf.constant([1.0, 2.0])
       # 'fetches' can be a singleton
       v = session.run(a)
       # v is the numpy array [10, 20]
       # 'fetches' can be a list.
       v = session.run([a, b])
       # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
       # 1-D array [1.0, 2.0]
       # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
       MyData = collections.namedtuple('MyData', ['a', 'b'])
       v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
       # v is a dict with
       # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
       # 'b' (the numpy array [1.0, 2.0])
       # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
     # [10, 20].
    ```
    
    The optional `feed_dict` argument allows the caller to override
    the value of tensors in the graph. Each key in `feed_dict` can be
    one of the following types:
    
    * If the key is a `tf.Tensor`, the
      value may be a Python scalar, string, list, or numpy ndarray
      that can be converted to the same `dtype` as that
      tensor. Additionally, if the key is a
      `tf.compat.v1.placeholder`, the shape of
      the value will be checked for compatibility with the placeholder.
    * If the key is a
      `tf.SparseTensor`,
      the value should be a
      `tf.compat.v1.SparseTensorValue`.
    * If the key is a nested tuple of `Tensor`s or `SparseTensor`s, the value
      should be a nested tuple with the same structure that maps to their
      corresponding values as above.
    
    Each value in `feed_dict` must be convertible to a numpy array of the dtype
    of the corresponding key.
    
    The optional `options` argument expects a [`RunOptions`] proto. The options
    allow controlling the behavior of this particular step (e.g. turning tracing
    on).
    
    The optional `run_metadata` argument expects a [`RunMetadata`] proto. When
    appropriate, the non-Tensor output of this step will be collected there. For
    example, when users turn on tracing in `options`, the profiled info will be
    collected into this argument and passed back.
    
    Args:
      fetches: A single graph element, a list of graph elements, or a dictionary
        whose values are graph elements or lists of graph elements (described
        above).
      feed_dict: A dictionary that maps graph elements to values (described
        above).
      options: A [`RunOptions`] protocol buffer
      run_metadata: A [`RunMetadata`] protocol buffer
    
    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list, or a dictionary with the
      same keys as `fetches` if that is a dictionary (described above).
      Order in which `fetches` operations are evaluated inside the call
      is undefined.
    
    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      ValueError: If `fetches` or `feed_dict` keys are invalid or refer to a
        `Tensor` that doesn't exist.


help(tf.contrib.graph_editor.swap_ts)

Help on function swap_ts in module tensorflow.contrib.graph_editor.reroute:

swap_ts(ts0, ts1, can_modify=None, cannot_modify=None)
    For each tensor's pair, swap the end of (t0,t1).
    
        B0 B1     B0 B1
        |  |    =>  X
        A0 A1     A0 A1
    
    Args:
      ts0: an object convertible to a list of `tf.Tensor`.
      ts1: an object convertible to a list of `tf.Tensor`.
      can_modify: iterable of operations which can be modified. Any operation
        outside within_ops will be left untouched by this function.
      cannot_modify: iterable of operations which cannot be modified.
        Any operation within cannot_modify will be left untouched by this
        function.
    Returns:
      The number of individual modifications made by the function.
    Raises:
      TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
      TypeError: if can_modify or cannot_modify is not None and cannot be
        converted to a list of tf.Operation.

Q: what does 'tf.contrib.graph_editor.swap_ts' function do?
The `tf.contrib.graph_editor.swap_ts` function is part of the TensorFlow's contrib module, which includes experimental and unsorted features that are not guaranteed to be stable. Specifically, `tf.contrib.graph_editor.swap_ts` is used to swap tensors in a computational graph. This can be useful for modifying or optimizing the graph by altering connections between operations.

Here's a brief explanation of its functionality:

- **Input tensors**: The function takes two lists of tensors as inputs.
- **Output**: It swaps the respective tensors in the computational graph.

Usage:

- This function allows for swapping the output of one tensor with another tensor, which can be helpful for graph editing and manipulation tasks.

Below is a rough guide on how you might use it:

```python
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge

# Assume you have two tensors `tensor_a` and `tensor_b` in a graph
tensor_a = ...
tensor_b = ...

# Swap the tensors in the graph
ge.swap_ts([tensor_a], [tensor_b])
```

Keep in mind that:

- The `tf.contrib` module has been deprecated in TensorFlow 2.x. If you are using TensorFlow 2.x, you should look for alternatives in the TensorFlow Addons or directly manipulate the graph using other utilities provided by TensorFlow 2.x.
- TensorFlow 2.x encourages using the `tf.function` API for graph-based computations rather than directly manipulating graphs.

If you are using TensorFlow 1.x, ensure that your code is compatible and be aware that future versions of TensorFlow might not support `tf.contrib`.

For TensorFlow 2.x, you may consider using `tf.Graph().as_default()` and related graph manipulation techniques, or use high-level APIs designed for model manipulation.

Q: in tensorflow computation graph, a graph node stands for an op, and an edge for data flow, so this function change the data flow in a graph?

Yes, `tf.contrib.graph_editor.swap_ts` essentially changes the data flow within a TensorFlow computation graph. Here’s a more detailed breakdown:

- **Graph Node (Op)**: In the TensorFlow computation graph, each node represents an operation (op), which could be anything from basic arithmetic operations to complex functions like matrix multiplications.
- **Edge (Tensor/Data Flow)**: The edges of the graph represent the flow of data (tensors) between these operations. An edge connects the output of one operation to the input of another operation.

When you use the `swap_ts` function, what happens is that the tensors specified in the input lists are swapped. This means:

1. **Redirect Outputs**: The tensor outputs from one operation are redirected to be the outputs of another operation.
2. **Change Connections**: This alters the connections between different operations, effectively modifying the flow of data in the graph.

### Example Illustration

Consider a simple graph with operations `A` and `B`, and tensors `a` and `b` as their respective outputs.

- Before swapping:
  - Operation `A` produces tensor `a`
  - Operation `B` produces tensor `b`

```plaintext
A -> a
B -> b
```

- After swapping `a` and `b` using `swap_ts`:

```python
ge.swap_ts([a], [b])
```

- The graph’s connections will be altered:

```plaintext
A -> b
B -> a
```

### Practical Implications

- **Graph Modification**: This allows you to modify the graph’s structure programmatically. Whether it’s for optimization, debugging, or altering the computation workflow, you have the flexibility to reroute data.
- **Dynamic Changes**: You can make these changes dynamically without needing to reconstruct the entire graph manually, saving time and effort.

### Caution

- **Compatibility**: Since `tf.contrib` is deprecated, ensure you are using TensorFlow 1.x for this function. In TensorFlow 2.x, you might need to rely on other methods for graph editing, such as `tf.function` and custom graph manipulation techniques.
- **Graph Integrity**: Be careful when altering the graph as unintended changes might lead to incorrect computations or runtime errors.

### Example Code

Here is a code snippet demonstrating the `swap_ts` function:

```python
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge

# Define two simple operations
a = tf.constant(1.0, name="a")
b = tf.constant(2.0, name="b")

# Swap the tensors in the graph
ge.swap_ts([a], [b])

# Validate the graph connections
with tf.Session() as sess:
    print(sess.run(a))  # should print 2.0
    print(sess.run(b))  # should print 1.0
```

This snippet swaps the outputs of constants `a` and `b`. The result of this swap will change the flow of data such that `a` outputs the value generated by `b` and vice versa.

Q: your demo code doesn't run as you specified, `sess.run(a)` still outputs 1.0

You're right, my previous example might have given an incorrect impression about how the `tf.contrib.graph_editor.swap_ts` function operates. Let's correct that and provide a more accurate usage.

The `swap_ts` function is intended to swap tensor outputs of one operation with another, which means it changes how the tensors are used in subsequent operations, rather than changing the tensor values directly.

Here's a more appropriate example demonstrating the manipulation of data flow with `swap_ts`:

```python
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge

# Create a simple graph
a = tf.constant(1.0, name="a")
b = tf.constant(2.0, name="b")

# Operations using tensors a and b
c = a + 1  # will be affected because it uses tensor a
d = b + 1  # will be affected because it uses tensor b

# Swap tensors a and b in the graph
ge.swap_ts([a], [b])

# Initialize the session and run operations
with tf.Session() as sess:
    print(sess.run(c))  # will be influenced by tensor b's value, expected output will be 3.0
    print(sess.run(d))  # will be influenced by tensor a's value, expected output will be 2.0
```

### Explanation

1. **Tensor Definition**: `a` and `b` are constant tensors.
2. **Operations Involving Tensors**:
   - `c` is defined to be `a + 1`
   - `d` is defined to be `b + 1`
3. **Swap Operation**: `ge.swap_ts([a], [b])` swaps the usage of tensor `a` with tensor `b`.
4. **Session Execution**:
   - Running `c` will now be influenced by the value from `b`
   - Running `d` will now be influenced by the value from `a`

As a result, since `c` originally depended on `a`, it will now yield results based on the value of `b` (i.e., `2.0 + 1 = 3.0`). Similarly, `d`, which originally depended on `b`, will now yield results based on the value of `a` (i.e., `1.0 + 1 = 2.0`).

### Note on Deprecation

Given that TensorFlow 1.x and `tf.contrib` are deprecated, you may want to consider transitioning to TensorFlow 2.x where you use `tf.function` along with other graph manipulation utilities.

TensorFlow 2.x focuses on eager execution, but retains the capability to define and manipulate graphs via `tf.function`. For more complex graph manipulations, consider exploring TensorFlow Addons or other third-party libraries.
```