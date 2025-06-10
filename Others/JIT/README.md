In TensorFlow, **"JIT coding"** refers to **Just-In-Time (JIT) compilation**, which is primarily achieved using `@tf.function`.

---

## 🚀 What is JIT in TensorFlow?

**JIT compilation** in TensorFlow:

* Converts your **Python functions** into **highly optimized TensorFlow computation graphs**.
* Achieved using the **`@tf.function` decorator**.
* This is TensorFlow’s version of "graph mode" execution — it's faster and more deployable.

---

## 🔧 How to Use JIT (`@tf.function`)

```python
import tensorflow as tf

@tf.function  # JIT compilation applied here
def my_model(x):
    return x * x + 2

# Running
x = tf.constant(5.0)
print(my_model(x))  # Fast execution via compiled graph
```

---

## 🧠 Why Use `@tf.function`?

| Feature                                 | Without `@tf.function` | With `@tf.function` |
| --------------------------------------- | ---------------------- | ------------------- |
| Mode                                    | Eager execution        | Graph execution     |
| Performance                             | Slower                 | Faster              |
| Deployability (e.g. TFLite, TF-Serving) | ❌                      | ✅                   |
| Optimizations (XLA, fusion)             | ❌                      | ✅                   |

---

## ⚡ JIT + XLA (Accelerated JIT)

For **even more performance**, you can combine JIT with **XLA (Accelerated Linear Algebra)**:

```python
@tf.function(jit_compile=True)
def xla_model(x, y):
    return tf.matmul(x, y)
```

* `jit_compile=True` enables **XLA**, which fuses operations and reduces kernel calls.
* Especially useful for **TPU/GPU performance optimization**.

---

## ✅ Summary

* **JIT coding** in TensorFlow means using `@tf.function` to **compile Python functions into TensorFlow graphs**.
* This results in **faster, portable, and optimized code**.
* Essential for deployment, performance-critical models, and inference on mobile/embedded systems.

Would you like a real example comparing performance of eager vs JIT compiled functions?

---

Below is a **self-contained benchmark** that measures how long a simple matrix-multiplication loop takes in

1. **Eager mode** (plain Python)
2. **Graph mode** (`@tf.function`, JIT compiled)
3. **Graph + XLA** (`@tf.function(jit_compile=True)`)

> **Run this in a fresh Python session** with TensorFlow ≥ 2.13 installed and a CPU or GPU backend.
> On a modern laptop CPU you’ll usually see the graph version run **≈ 2× faster** than eager; with a recent GPU or Apple Silicon the speed-up can be much larger.

```python
import tensorflow as tf
import time
import numpy as np
tf.keras.backend.clear_session()   # clean slate

# ------------ 1. Build some random data ------------
N = 1_000          # number of iterations for the test loop
d = 1024           # square matrix dimension

rng = np.random.default_rng(42)
A = tf.constant(rng.standard_normal((d, d), dtype=np.float32))
B = tf.constant(rng.standard_normal((d, d), dtype=np.float32))

# ------------ 2. Define three versions of the same op ------------

def matmul_eager(a, b):
    return tf.matmul(a, b)

@tf.function            # JIT / graph mode
def matmul_graph(a, b):
    return tf.matmul(a, b)

@tf.function(jit_compile=True)   # Graph + XLA fusion
def matmul_xla(a, b):
    return tf.matmul(a, b)

# ------------ 3. Warm-up (always do this!) ------------
matmul_eager(A, B)
matmul_graph(A, B)
matmul_xla(A, B)

# ------------ 4. Timing helper ------------
def time_it(fn, name):
    start = time.perf_counter()
    for _ in range(N):
        _ = fn(A, B)
    tf.keras.backend.clear_session()  # avoid memory blow-up
    elapsed = time.perf_counter() - start
    print(f"{name:<18}: {elapsed*1e3/N:7.2f} ms / call")

# ------------ 5. Run the benchmark ------------
time_it(matmul_eager, "Eager")
time_it(matmul_graph, "Graph (@tf.function)")
time_it(matmul_xla,  "Graph + XLA")

```

### 🔍 What to expect

| Mode                       | Typical CPU time\* | Explanation                                                      |
| -------------------------- | ------------------ | ---------------------------------------------------------------- |
| **Eager**                  | \~4–6 ms / call    | Every `tf.matmul` launches a separate kernel from Python.        |
| **Graph (`@tf.function`)** | \~2–3 ms / call    | Python is bypassed; TensorFlow executes a compiled graph.        |
| **Graph + XLA**            | \~1.5–2 ms / call  | XLA fuses ops, reduces memory traffic, and chooses fast kernels. |

\*Numbers vary by hardware; on GPUs the absolute times shrink, but the **relative speed-ups are similar or larger**.

### Key takeaways

* **`@tf.function`** is TensorFlow’s *JIT switch*: one line can give you 2-10× speed-ups and makes the model exportable (SavedModel, TFLite, TF-Serving, SNPE, etc.).
* **`jit_compile=True`** hands the graph to XLA for deeper fusion and kernel selection. It can give further gains, especially on TPUs or when many small ops can be fused.
* Always **warm-up before timing** to let TensorFlow build/optimize the graph and cache kernels.

Feel free to tweak `N`, `d`, or replace `tf.matmul` with a more complex model to see how JIT affects real workloads.
