#include "leetcode.h"

using namespace std;

/*
leetcode: 677

Implement a MapSum class with insert, and sum methods.

For the method insert, you'll be given a pair of (string, integer). The string represents the key and the integer represents the value.
If the key already existed, then the original key-value pair will be overridden to the new one.
For the method sum, you'll be given a string representing the prefix, and you need to return the sum of all the pairs' value whose key starts with the prefix.
you may assume that key and prefix consist of only lowercase English letters a-z.

Example 1:
    Input: insert("apple", 3), Output: Null
    Input: sum("ap"), Output: 3
    Input: insert("app", 2), Output: Null
    Input: sum("ap"), Output: 5
*/

struct MapNode {
    int val;
    bool is_leaf;
    std::vector<MapNode*> children;
    MapNode(int n=0):
        val(n),
        is_leaf(false) {
        children.resize(128, nullptr);
    }
    ~MapNode() {
        for (auto n: children) {
            delete n;
        }
    }
};

class MapSum {
public:
    void insert(const string& key, int val);
    int sum(const string& prefix);
private:
    MapNode m_root;
};


void MapSum::insert(const string& key, int val) {
    MapNode* p = &m_root;
    for (auto c: key) {
        if (p->children[c] == nullptr) { // create a node if it doesn't exist yet
            p->children[c] = new MapNode;
        }
        p = p->children[c];
    }
    // set node value
    p->is_leaf = true;
    p->val = val;
}


int MapSum::sum(const string& prefix) {
    if (prefix.empty()) { // trivial case
        return 0;
    }
    // find the enter point
    MapNode* p = &m_root;
    for (auto c: prefix) {
        p = p->children[c];
        if (p == nullptr) {
            return 0;
        }
    }
    // traverse from the enter pointer, and sum values of all leaf nodes
    // return the sum of subtree rooted at node
    function<int(MapNode*)> dfs = [&] (MapNode* node) {
        if (node == nullptr) { //trivial case
            return 0;
        }
        int ans = 0;
        if (node->is_leaf) { // sum up the values of leaf node
            ans += node->val;
        }
        for (auto c: node->children) {
            ans += dfs(c);
        }
        return ans;
    };
    return dfs(p);
}


void MapSum_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    MapSum tm;
    int n = (int)funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "insert") {
            tm.insert(funcArgs[i][0], std::stoi(funcArgs[i][1]));
            SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
        } else if (funcOperations[i] == "sum") {
            int actual = tm.sum(funcArgs[i][0]);
            if (actual == std::stoi(ans[i])) {
                SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
            } else {
                SPDLOG_ERROR("{}({}) failed. Expected={}, actual={}", funcOperations[i], funcArgs[i][0], ans[i], actual);
            }
        }
    }
}


int main() {
    SPDLOG_WARN("Running MapSum tests:");
    TIMER_START(MapSum);
    MapSum_scaffold(
        "[MapSum,insert,sum,insert,sum,insert,sum,sum,sum]",
        "[[],[apple,3],[ap],[apm,2],[ap],[approve,7],[ap],[app],[bo]]",
        "[null,null,3,null,5,null,12,10,0]");
    TIMER_STOP(MapSum);
    SPDLOG_WARN("MapSum tests use {} ms", TIMER_MSEC(MapSum));
}
