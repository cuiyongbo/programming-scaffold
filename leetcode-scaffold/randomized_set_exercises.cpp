#include "leetcode.h"

using namespace std;

/*
Implement the RandomizedSet class:
- RandomizedSet() Initializes the RandomizedSet object.
- bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
- bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
- int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.

You must implement the functions of the class such that each function works in average O(1) time complexity.

Example 1:

Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.
 
Constraints:

-231 <= val <= 231 - 1
At most 2 * 105 calls will be made to insert, remove, and getRandom.
There will be at least one element in the data structure when getRandom is called.
*/

class RandomizedSet {
public:
    RandomizedSet() {
    }

    bool insert(int val) {
        if (m_loc_map.count(val)) {
            return false;
        }
        m_loc_map[val] = m_datastore.size();
        m_datastore.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (!m_loc_map.count(val)) {
            return false;
        }
        int i = m_loc_map[val];
        // update the index of the last element
        m_datastore[i] = m_datastore.back();
        m_loc_map[m_datastore.back()] = i;
        // remove val
        m_datastore.pop_back();
        m_loc_map.erase(val);
        return true;
    }

    int getRandom() {
        return m_datastore[rand() % m_datastore.size()];
    }

private:
    std::unordered_map<int, int> m_loc_map; // element, element index in m_datastore
    std::vector<int> m_datastore;
};


void RandomizedSet_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    RandomizedSet tm;
    int n = (int)funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "insert") {
            bool actual = tm.insert(std::stoi(funcArgs[i][0]));
            string actual_str = actual ? "true" : "false";
            if (actual_str == ans[i]) {
                SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
            } else {
                SPDLOG_ERROR("{}({}) failed, expectedResult={}, actual={}", funcOperations[i], funcArgs[i][0], ans[i], actual);
            }
        } else if (funcOperations[i] == "remove") {
            bool actual = tm.remove(std::stoi(funcArgs[i][0]));
            string actual_str = actual ? "true" : "false";
            if (actual_str == ans[i]) {
                SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
            } else {
                SPDLOG_ERROR("{}({}) failed, expectedResult={}, actual={}", funcOperations[i], funcArgs[i][0], ans[i], actual);
            }
        }
    }
}


int main() {
    SPDLOG_WARN("Running RandomizedSet tests:");
    TIMER_START(RandomizedSet);
    RandomizedSet_scaffold(
        "[RandomizedSet, insert, remove, insert, getRandom, remove, insert, getRandom]",
        "[[], [1], [2], [2], [], [1], [2], []]",
        "[null, true, false, true, 2, true, false, 2]");
    TIMER_STOP(RandomizedSet);
    SPDLOG_WARN("RandomizedSet tests use {} ms", TIMER_MSEC(RandomizedSet));
}
