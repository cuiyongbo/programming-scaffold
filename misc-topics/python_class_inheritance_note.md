# In python, I have two class A, B, and B inherits from A, A has a method called foo which is overrided in B, and I want call `A.foo` in B's methods, how to do it?

你这个问题是关于 Python 中子类如何调用父类被重写的方法。非常经典，下面详细说明几种常见写法。

---

## 场景描述

```python
class A:
    def foo(self):
        print("A.foo")

class B(A):
    def foo(self):
        print("B.foo")
        # 这里想调用 A.foo
```

---

## 1. 使用 `super()` 调用父类方法（推荐）

```python
class B(A):
    def foo(self):
        print("B.foo")
        super().foo()  # 调用父类 A 的 foo
```

**说明：**

- `super()` 返回父类对象，调用其方法。
- 推荐写法，支持多继承，语义清晰。
- Python 3 中可以直接写 `super().foo()`，不需要传参数。

---

## 2. 直接通过类名调用父类方法

```python
class B(A):
    def foo(self):
        print("B.foo")
        A.foo(self)  # 直接调用父类方法，手动传 self
```

**说明：**

- 直接用类名调用父类方法，需要手动传递当前实例 `self`。
- 适合单继承，简单明了。
- 多继承时不推荐，因为可能绕过 MRO（方法解析顺序）。

---

## 3. 示例完整代码及调用

```python
class A:
    def foo(self):
        print("A.foo")

class B(A):
    def foo(self):
        print("B.foo")
        super().foo()   # 或者 A.foo(self)

b = B()
b.foo()
```

输出：

```
B.foo
A.foo
```

---

## 总结

| 方法            | 说明                          | 推荐度      |
|-----------------|-------------------------------|-------------|
| `super().foo()` | 支持多继承，语义清晰           | ★★★★★       |
| `A.foo(self)`   | 简单直接，单继承时可用         | ★★★☆☆       |

---

如果你需要，我可以帮你写多继承场景下的调用示例，或者解释 `super()` 的工作原理！

# what if A inherits from another class C, and A override foo method in C, then how to call `C.foo` in B's methods?

Great question! When you have a class hierarchy like:

```
C
↓
A (overrides foo)
↓
B (overrides foo)
```

and you want to call `C.foo` inside `B`'s method (bypassing `A.foo`), there are several ways to do it.

---

## Scenario

```python
class C:
    def foo(self):
        print("C.foo")

class A(C):
    def foo(self):
        print("A.foo")

class B(A):
    def foo(self):
        print("B.foo")
        # How to call C.foo here?
```

---

## How to call `C.foo` in `B.foo`?

### 1. Direct call using class name (recommended for this case)

```python
class B(A):
    def foo(self):
        print("B.foo")
        C.foo(self)  # Directly call C.foo, pass self explicitly
```

- This calls `C.foo` directly, **bypassing A.foo**.
- You must pass `self` explicitly because it's an unbound method.

---

### 2. Using `super()` with arguments (less common, but possible)

You can use `super()` to skip a class in the MRO by specifying the class and instance explicitly:

```python
class B(A):
    def foo(self):
        print("B.foo")
        super(A, self).foo()  # Calls the next foo after A in MRO, which is C.foo
```

- `super(A, self)` means: "start looking for methods **after** class A in the MRO of `self`".
- This will call `C.foo` because `C` is next after `A` in the MRO.

---

### 3. Explanation of MRO

- The MRO (method resolution order) for `B` is `[B, A, C, object]`.
- `super()` without arguments in `B` looks for the next method **after B**, which is `A.foo`.
- `super(A, self)` looks for the next method **after A**, which is `C.foo`.

---

## Complete example

```python
class C:
    def foo(self):
        print("C.foo")

class A(C):
    def foo(self):
        print("A.foo")

class B(A):
    def foo(self):
        print("B.foo")
        print("Calling C.foo directly:")
        C.foo(self)

        print("Calling C.foo via super():")
        super(A, self).foo()

b = B()
b.foo()
```

### Output

```
B.foo
Calling C.foo directly:
C.foo
Calling C.foo via super():
C.foo
```

---

## Summary

| Method                         | Description                              | Usage scenario                        |
|--------------------------------|------------------------------------------|-------------------------------------|
| `C.foo(self)`                  | Direct call, bypasses `A.foo`            | Simple, explicit call                |
| `super(A, self).foo()`         | Call next method after `A` in MRO (`C.foo`) | More dynamic, respects MRO           |

---

If you want, I can help explain more about Python’s MRO or how to use `super()` in complex multiple inheritance!


# why does not python support function overload?

Great question! Python **does not support traditional function overloading** (like C++ or Java) where multiple functions with the same name but different parameter types or counts coexist and the correct one is chosen at compile time. Here’s why, and how Python approaches similar problems:

---

## Why Python doesn’t support traditional function overloading

### 1. **Dynamic Typing**

- Python is a **dynamically typed language**: variables and function arguments don’t have fixed types known at compile time.
- Overloading based on argument types requires **static type information** to select the correct function version at compile time, which Python doesn’t have.

### 2. **Simple and Consistent Function Model**

- Python functions are **first-class objects** and can be redefined or replaced at runtime.
- Allowing multiple functions with the same name but different signatures would complicate the language and runtime.
- Instead, Python favors **simplicity and explicitness**.

### 3. **Flexible Function Signatures**

- Python functions support **default arguments**, `*args`, and `**kwargs`, allowing a single function to handle multiple calling patterns.
- This reduces the need for multiple overloads.

---

## How to achieve similar behavior in Python?

### 1. **Default arguments and variable arguments**

```python
def func(a, b=None):
    if b is None:
        print(f"One argument: {a}")
    else:
        print(f"Two arguments: {a}, {b}")
```

### 2. **Type checking inside functions**

```python
def func(x):
    if isinstance(x, int):
        print("int version")
    elif isinstance(x, str):
        print("str version")
```

### 3. **Using `functools.singledispatch`**

- Python 3.4+ provides `functools.singledispatch` to do **function overloading based on the first argument’s type**.

Example:

```python
from functools import singledispatch

@singledispatch
def func(arg):
    print("Default implementation")

@func.register
def _(arg: int):
    print("Integer version")

@func.register
def _(arg: str):
    print("String version")

func(10)    # Integer version
func("hi")  # String version
func(3.14)  # Default implementation
```

---

## Summary

| Aspect                       | Explanation                              |
|------------------------------|------------------------------------------|
| Static overloading            | Not supported due to dynamic typing      |
| Python function model         | Single function object per name          |
| Alternative approaches       | Default args, type checks, `singledispatch` |
| Philosophy                   | Simplicity, explicitness over complexity |

---

If you want, I can help you write a Pythonic overload-like dispatcher or explain `singledispatch` in more detail!
