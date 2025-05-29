#include <iostream>
#include <memory>
 
class Base {
public:
    int a;
    virtual void f() const { std::cout << "I am base!\n"; }
    virtual ~Base() {}
};
 
class Derived : public Base {
public:
    void f() const override { std::cout << "I am derived!\n"; }
    ~Derived() {}
};

// https://en.cppreference.com/w/cpp/memory/shared_ptr/pointer_cast
// https://stackoverflow.com/questions/34048692/stdstatic-pointer-cast-vs-static-caststdshared-ptra
int main() {
    auto basePtr = std::make_shared<Base>();
    std::cout << "Base pointer says: ";
    basePtr->f();
 
    auto derivedPtr = std::make_shared<Derived>();
    std::cout << "Derived pointer says: ";
    derivedPtr->f();
 
    // static_pointer_cast to go up class hierarchy
    basePtr = std::static_pointer_cast<Base>(derivedPtr);
    std::cout << "Base pointer to derived says: ";
    basePtr->f();
 
    // dynamic_pointer_cast to go down/across class hierarchy
    auto downcastedPtr = std::dynamic_pointer_cast<Derived>(basePtr);
    if (downcastedPtr)
    {
        std::cout << "Downcasted pointer says: ";
        downcastedPtr->f();
    }
 
    // All pointers to derived share ownership
    std::cout << "Pointers to underlying derived: "
              << derivedPtr.use_count() << ", "
              << downcastedPtr.use_count() << ", "
              << basePtr.use_count()
              << '\n';
}