#include <iostream>
#include <mutex>
#include <shared_mutex>
//#include <syncstream>
#include <thread>

// g++ std_shared_mutex_demo.cpp -std=c++17
class ThreadSafeCounter {
public:
  ThreadSafeCounter() = default;

  int get() const {
    // lock mutex in shared mode
    std::shared_lock lock(m_mutex);
    return m_value;
  }

  void increment() {
    // lock mutex in exclusive mode
    std::unique_lock lock(m_mutex);
    m_value++;
  }

  void reset() {
    // lock mutex in exclusive mode
    std::unique_lock lock(m_mutex);
    m_value = 0;
  }

private:
  mutable std::shared_mutex m_mutex; // to use m_mutex in const member function
  int m_value;
};

int main() {
  ThreadSafeCounter counter;
  auto increment_and_print = [&counter] () {
    for (int i=0; i<2; i++) {
      counter.increment();
      //std::osyncstream(std::cout) << std::this_thread::get_id() << " " << counter.get() << '\n';
      std::cout << std::this_thread::get_id() << " " << counter.get() << '\n';
    }
  };
  std::thread thread1(increment_and_print);
  std::thread thread2(increment_and_print);
  thread1.join();
  thread2.join();
}