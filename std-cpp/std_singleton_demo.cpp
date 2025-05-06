#include <iostream>

using namespace std;

// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
class S {
public:
    static S& getInstance()
    {
        static S    instance; // Guaranteed to be destroyed. Instantiated on first use.
        return instance;
    }
private:
    S() {}                    // Constructor? (the {} brackets) are needed here.

    // C++ 03
    // ========
    // Don't forget to declare these two. You want to make sure they
    // are inaccessible(especially from outside), otherwise, you may accidentally get copies of
    // your singleton appearing.
    //S(S const&);              // Don't Implement
    //void operator=(S const&); // Don't implement

    // C++ 11
    // =======
    // We can use the better technique of deleting the methods
    // we don't want.
public:
    S(S const&)               = delete;
    void operator=(S const&)  = delete;

    // Note: Scott Meyers mentions in his Effective Modern
    //       C++ book, that deleted functions should generally
    //       be public as it results in better error messages
    //       due to the compilers behavior to check accessibility
    //       before deleted status
};

class Singleton {
public:
    static Singleton& getInstance() {
        // thread-safe in c++11
        /*
        The C++ standard guarantees that a static local variable is initialized only once, even if called from multiple threads, thus avoiding race conditions during their initialization phase.
        */
        static Singleton instance;
        return instance;
    }

    // disable copy
    Singleton(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(const Singleton&) = delete;

private:
    Singleton() {}

};

int main() {
    Singleton& s1 = Singleton::getInstance();
    Singleton& s2 = Singleton::getInstance();
    cout << "s1: " << &s1 << "\n";
    cout << "s2: " << &s2 << "\n";
    //Singleton  ss;
    return 0;
}
