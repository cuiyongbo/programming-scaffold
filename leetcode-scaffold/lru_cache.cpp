#include <iostream>
#include <map>
#include <unordered_map>
#include <list>
#include <cassert>
#include <mutex>
#include <shared_mutex>

using namespace std;

/*
Leetcode 146:

Design a data structure that follows the constraints of a Least Recently Used (LRUCache) cache.

Implement the LRUCache class:

`LRUCache(int capacity)` Initialize the LRUCache cache with positive size capacity.

`int get(int key)` Return the value of the key if the key exists, otherwise return -1.

`void put(int key, int value)` Update the value of the key if the key exists. 
Otherwise, add the key-value pair to the cache. If the number of keys exceeds 
the capacity from this operation, evict the least recently used key.

Follow up: Could you do get and put in O(1) time complexity?

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
    LRUCache lRUCache = new LRUCache(2);
    lRUCache.put(1, 1); // cache is {1=1}
    lRUCache.put(2, 2); // cache is {1=1, 2=2}
    lRUCache.get(1);    // return 1
    lRUCache.put(3, 3); // LRUCache key was 2, evicts key 2, cache is {1=1, 3=3}
    lRUCache.get(2);    // returns -1 (not found)
    lRUCache.put(4, 4); // LRUCache key was 1, evicts key 1, cache is {4=4, 3=3}
    lRUCache.get(1);    // return -1 (not found)
    lRUCache.get(3);    // return 3
    lRUCache.get(4);    // return 4
*/

namespace std_implementation {

class LRUCache {
private:
    std::mutex m_mutex;
    int m_capacity;
    std::list<std::pair<int, int>> m_nodes; // <key, value>
    std::map<int, list<std::pair<int, int>>::iterator> m_node_map; // key, position in the list 
public:
    LRUCache(int cap) { m_capacity = cap; }
    ~LRUCache() {}
    int get(int key);
    void put(int key, int value);
    void put_without_lock(int key, int value);
    void display();
};

void LRUCache::put(int key, int value) {
    std::unique_lock<std::mutex> guard(m_mutex);
    put_without_lock(key, value);
}

void LRUCache::put_without_lock(int key, int value) {
    if (m_node_map.count(key)) { // key already exists
        // first remove key from list
        m_nodes.erase(m_node_map[key]);
        // then from map
        m_node_map.erase(key);
    } else { // key doesn't exist
        // remove the oldest element if capacity reaches
        if ((int)m_nodes.size() == m_capacity) {
            auto b = m_nodes.back();
            m_node_map.erase(b.first); // first remove iterator from the map
            m_nodes.pop_back(); // then remove iterator from the list
        }
    }
    m_nodes.push_front(std::make_pair(key, value)); // move <key, value> to the front of the list
    m_node_map[key] = m_nodes.begin(); // update map
}

int LRUCache::get(int key) {
    std::unique_lock<std::mutex> guard(m_mutex);
    auto it = m_node_map.find(key);
    if (it == m_node_map.end()) {
        return -1;
    }
    // you need to change the position of key in the list
    /*
    Transfers elements from one list to another.
    No elements are copied or moved, only the internal pointers of the list nodes are re-pointed.
    No iterators or references become invalidated, the iterators to moved elements remain valid, but now refer into *this, not into other.
    */
    // void splice(const_iterator pos, list& other, const_iterator it);
    // Transfers the element pointed to by `it` from other into *this. The element is inserted before the element pointed to by `pos`.
    m_nodes.splice(m_nodes.begin(), m_nodes, it->second); // move key to the front of m_nodes
    return it->second->second;
}

void LRUCache::display() {
    std::unique_lock<std::mutex> guard(m_mutex);
    for (auto it: m_nodes) {
        printf("(%d,%d)", it.first, it.second);
    }
    printf("\n");
}

}


int main() {
    using std_implementation::LRUCache;
    //using thread_safe_implementation::LRUCache;
    LRUCache lru(3);
    int p = lru.get(1);
    assert(p == -1);
    lru.put(1, 10);
    lru.put(2, 20);
    p = lru.get(1);
    assert(p == 10);
    lru.display(); // (1,10)(2,20)
    lru.put(3, 30);
    lru.put(4, 40); // (2,20) would be evicted
    lru.display();  // (4,40)(3,30)(1,10)
    lru.put(2, 22); // (1,10) would be evicted
    lru.display();  // (2,22)(4,40)(3,30)
    p = lru.get(2);
    assert(p == 22);
    lru.display();  // (2,22)(4,40)(3,30)
    p = lru.get(2);
    assert(p == 22);
    assert(p == 2);
    lru.display();  // (2,22)(4,40)(3,30)
    cout << p << endl;
}
