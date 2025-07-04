#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;

/* 
leetcode: 208 
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:
    Trie() Initializes the trie object.
    void insert(String word) Inserts the string `word` into the trie.
    boolean search(String word) Returns true if the string `word` is in the trie (i.e., was inserted before), and false otherwise.
    boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix `prefix`, and false otherwise.
*/

namespace naive_version {
class Trie {
struct TrieNode {
    bool is_end; // true if there is some word ending with current node
    map<char, TrieNode*> children; // nodes after current nodes
    TrieNode(): is_end(false) {}
    ~TrieNode() {
        for (auto t: children) {
            delete t.second;
        }
    }
};
private:
    TrieNode m_root; // root of trie tree

public:
    void insert(string word) {
        TrieNode* p = &m_root; // start from the root
        for (auto c: word) {
            if (p->children.count(c) == 0) { // create node if it doesn't exist yes
                p->children[c] = new TrieNode;
            }
            p = p->children[c]; // go to the next node
        }
        p->is_end = true; // mark the end of word
    }
    
    bool search(string word) {
        TrieNode* p = &m_root;
        for (auto c: word) {
            if (p->children.count(c) == 0) { // every node for the character in the word must exist
                return false;
            }
            p = p->children[c];
        }
        return p->is_end; // it has to be a leaf node
    }
    
    bool startsWith(string prefix) {
        TrieNode* p = &m_root;
        for (auto c: prefix) {
            if (p->children.count(c) == 0) {
                return false;
            }
            p = p->children[c];
        }
        return true;        
    }
};

}


void TrieTree_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    TrieTree tm;
    //naive_version::Trie tm;
    int n = (int)funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "insert") {
            tm.insert(funcArgs[i][0]);
            SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
        } else if (funcOperations[i] == "search" || funcOperations[i] == "startsWith") {
            bool actual = funcOperations[i] == "search" ? 
                                tm.search(funcArgs[i][0]) :
                                tm.startsWith(funcArgs[i][0]);
            string actual_str = actual ? "true" : "false";
            if (actual_str == ans[i]) {
                SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
            } else {
                SPDLOG_ERROR("{}({}) failed, expectedResult={}, actual={}", funcOperations[i], funcArgs[i][0], ans[i], actual);
            }
        }
    }
}

/*
leetcode: 386 
Given an integer n, return all the numbers in the range [1, n] sorted in lexicographical order.
You must write an algorithm that runs in O(n) time and uses O(1) extra space. 

Example 1:

Input: n = 13
Output: [1,10,11,12,13,2,3,4,5,6,7,8,9]
Example 2:

Input: n = 2
Output: [1,2]

*/

class Solution {
public:
    vector<int> lexicalOrder(int n) {
        vector<int> ans; ans.reserve(n);
        std::function<void(int)> dfs = [&] (int cur) {
            if (cur > n) { // termination
                return;
            }
            ans.push_back(cur);
            // at each level traverse each digit in ascending order
            for (int i=0; i<=9; i++) {
                dfs(cur*10 + i);
            }
        };
        // NOTE that MSB can't be zero
        for (int i=1; i<=9; i++) {
            dfs(i);
        }
        return ans;
    }
};


void lexicalOrder_scaffold(int n, const string& expected) {
    Solution s;
    // initial value of reference to non-const must be an lvalueC/C++(461)
    const vector<int>& actual = s.lexicalOrder(n);
    const vector<int>& ans = stringTo1DArray<int>(expected);
    if (actual == ans) {
        SPDLOG_INFO("case({}) passed", n);
    } else {
        SPDLOG_ERROR("case({}) failed, expectedResult={}, actual={}", n, expected, numberVectorToString(actual));
    }
}


int main() {
    SPDLOG_WARN("Running TrieTree tests:");
    TIMER_START(TrieTree);
    TrieTree_scaffold(
        "[TrieTree,insert,insert,insert,search,search,startsWith,startsWith,startsWith]",
        "[[],[hello],[heros],[hell],[hello],[hero],[hero],[he],[hr]]",
        "[null,null,null,null,true,false,true,true,false]");
    TIMER_STOP(TrieTree);
    SPDLOG_WARN("TrieTree tests use {} ms", TIMER_MSEC(TrieTree));

    SPDLOG_WARN("Running lexicalOrder tests:");
    TIMER_START(lexicalOrder);
    lexicalOrder_scaffold(13, "[1,10,11,12,13,2,3,4,5,6,7,8,9]");
    lexicalOrder_scaffold(2, "[1,2]");
    TIMER_STOP(lexicalOrder);
    SPDLOG_WARN("lexicalOrder tests use {} ms", TIMER_MSEC(lexicalOrder));
}
