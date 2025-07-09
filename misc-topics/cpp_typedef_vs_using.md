# Difference Between using and typedef in C++

In C++, both using and typedef are used to create type aliases, but they have some differences in their usage and capabilities.

## typedef Keyword

The typedef keyword is used to create an alias for an existing data type, making it easier to refer to complex types with a simpler name. It is commonly used for aliasing standard data types, user-defined data types, and pointers.

Syntax: `typedef <existing_type> <alias_name>;`

Example:

```cpp
typedef std::vector<int> vInt;
vInt v; // Now, v is a vector of integers
```

Limitations:

- Templates: typedef cannot be used with templates. (使用 typedef 时被起别名的类型必须是确定的)
- Readability: It can be less readable and harder to modify.
- Pointer Declaration: The pointer declaration is not very clean.

Example with Struct:

```cpp
typedef struct {
    char title[50];
    char author[50];
    char subject[100];
    int book_id;
} Book;

Book myBook;
```

### typedef 是否可以结合模板使用

可以, 前提是运行到 typdef 时, 编译器可以确定原始类型的类型

- OK

```cpp
using namespace std;

template<typename T>
struct Salary {
typedef std::map<string, T> type;
};

Salary<float>::type Employee_Salary;

int main() {
    Employee_Salary["peter"] = 0.1;
}
```

- Wrong

```cpp
template<typename T>
typedef vector<T> arr_ptr_t; //  compilation error: a typedef cannot be a template

int main() {
    // ...
}
```

## using Keyword

The using keyword is more versatile and can be used to create type aliases, bring specific members into the current scope, and bring base class variables/methods into the derived class’s scope.

Syntax: `using <alias_name> = <existing_type>;`

Example:

```cpp
using vInt = std::vector<int>;
vInt v; // Now, v is a vector of integers
```

Advantages:

- Templates: using can be used with templates, making it more flexible.
- Readability: It is more readable and easier to modify.
- Pointer Declaration: The pointer declaration is cleaner.

Example with Templates:

```cpp
template<typename T>
using Salary = std::unordered_map<Employee_id, std::vector<T>>;

Salary<float> employeeSalary;
```

Example with Base Class Method:

```cpp
class Parent {
public:
    void add(int a, int b) {
        std::cout << "Result = " << a + b << std::endl;
    }
};

class Child : protected Parent {
public:
    using Parent::add;
};

int main() {
    Child obj;
    obj.add(15, 30); // Accessing Parent's add method
    return 0;
}
```

Conclusion

While both `typedef` and `using` serve the purpose of creating type aliases, `using` is generally preferred due to its flexibility, readability, and ability to work with templates. `typedef` is still useful in some contexts, but `using` provides a more modern and cleaner approach to type aliasing in C++

- [C++ typedef vs typedef](https://www.geeksforgeeks.org/cpp-using-vstypedef/)
