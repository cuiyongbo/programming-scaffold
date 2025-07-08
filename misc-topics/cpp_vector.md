# what is the difference between `std::vector::push_back` and `std::vector::emplace_back`?

`std::vector::push_back` and `std::vector::emplace_back` are both member functions used to add elements to the end of a `std::vector`, but they have key differences in how they achieve this.

### `std::vector::push_back`

**Usage:**
- `void push_back(const T& value);`
- `void push_back(T&& value);`

**Behavior:**
- `push_back` appends a copy (or move) of the element to the end of the vector.
- If the element is an lvalue, it is copied into the vector.
- If the element is an rvalue, it is moved into the vector.

**Example:**
```cpp
#include <vector>
#include <string>

int main() {
    std::vector<std::string> vec;
    std::string str = "hello";

    vec.push_back(str);        // Copies str into the vector
    vec.push_back("world");    // Constructs a temporary std::string and moves it into the vector

    return 0;
}
```

### `std::vector::emplace_back`

**Usage:**
- `template< class... Args > void emplace_back( Args&&... args );`

**Behavior:**
- `emplace_back` constructs the element in-place at the end of the vector using the provided arguments.
- It uses perfect forwarding to pass the arguments directly to the constructor of the element type.
- This avoids unnecessary copy or move operations.

**Example:**
```cpp
#include <vector>
#include <string>

int main() {
    std::vector<std::string> vec;

    vec.emplace_back("hello"); // Constructs "hello" directly in the vector
    vec.emplace_back(5, 'a');  // Constructs string "aaaaa" directly in the vector

    return 0;
}
```

### Key Differences

1. **Construction:**
   - `push_back` requires an already created object to be passed, which means it could incur a copy or move operation.
   - `emplace_back` constructs the object in place directly in the vector, using the arguments provided, potentially avoiding unnecessary copies or moves.

2. **Performance:**
   - `emplace_back` can be more efficient for complex types that are expensive to copy or move since it constructs the element in-place.
   - `push_back` may involve an extra copy or move which could be less efficient.

3. **Usage Syntax:**
   - `push_back` is used with an already created object.
   - `emplace_back` is used with the arguments for the constructor of the object.

### Example Comparison

Suppose you have a class with a non-trivial constructor:

```cpp
class MyClass {
public:
    MyClass(int x, double y) : x_(x), y_(y) {}
    // Other members...
private:
    int x_;
    double y_;
};
```

#### Using `push_back`:
```cpp
#include <vector>

int main() {
    std::vector<MyClass> vec;

    MyClass obj(10, 20.5);
    vec.push_back(obj);         // Calls the copy constructor
    vec.push_back(MyClass(10, 20.5)); // Calls the move constructor

    return 0;
}
```

#### Using `emplace_back`:
```cpp
#include <vector>

int main() {
    std::vector<MyClass> vec;

    vec.emplace_back(10, 20.5); // Constructs MyClass(10, 20.5) directly in place in the vector

    return 0;
}
```

### Summary

- **`push_back`:** Appends a copy (or move) of an existing object to the end of the vector.
- **`emplace_back`:** Constructs the object in place at the end of the vector using the provided arguments, potentially avoiding unnecessary copies or moves.

For simple types or when readability is a priority, `push_back` is often sufficient. For complex types or when optimizing for performance, especially to avoid unnecessary copies or moves, `emplace_back` is typically the better choice.
