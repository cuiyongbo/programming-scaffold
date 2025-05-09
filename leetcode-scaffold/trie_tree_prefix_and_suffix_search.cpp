#include "leetcode.h"
#include <string_view>

using namespace std;

/*
leetcode: 745

Given many words, words[i] has weight i. Design a class WordFilter that supports one function, WordFilter.filter(String prefix, String suffix). 
It will return the word with given prefix and suffix with maximum weight. If no word exists, return -1.

Examples:
    Input:
    WordFilter(["apple"])
    WordFilter.filter("a", "e") // returns 0
    WordFilter.filter("b", "") // returns -1
Note: 
    * `words[i]` has length in range `[1, 10]`.
    * `prefix, suffix` have lengths in range `[0, 10]`.
    * `words[i]` and `prefix, suffix` queries consist of lowercase letters only.
*/
class WordFilter {
struct FilterNode {
    FilterNode() {
        val = -1;
        is_leaf = false;
        children.assign(128, nullptr);
    }
    ~FilterNode() {
        for (auto n: children){
            delete n;
        }
    }
    int val;
    bool is_leaf;
    std::vector<FilterNode*> children;
};
public:
    void buildDict(const vector<string>& dict);
    int filter(const string& prefix, const string& suffix);
private:
    void insert(const string& key, int val);
private:
    FilterNode m_root;
};


void WordFilter::buildDict(const vector<string>& dict) {
    for (int i=0; i<(int)dict.size(); i++) {
        insert(dict[i], i);
    }
}


void WordFilter::insert(const string& key, int val){
    FilterNode* cur = &m_root; // start from root of the tree
    for (auto c: key) {
        if (cur->children[c] == nullptr) { // create node i
            cur->children[c] = new FilterNode;
        }
        cur = cur->children[c];
    }
    cur->val = val;
    cur->is_leaf = true;
}


int WordFilter::filter(const string& prefix, const string& suffix) {
    if (prefix.empty()) {
        return -1;
    }
    // find the enter point with prefix
    FilterNode* p = &m_root;
    for (auto c: prefix) {
        if (p->children[c] == nullptr) {
            return -1;
        }
        p = p->children[c];
    }
    // traverse from the enter pointer, finding the candidate with maximum weight
    int ans = -1;
    std::string candidate = prefix;
    function<void(FilterNode*)> backtrace = [&] (FilterNode* node) {
        if (node == nullptr) {
            return;
        }
        if (node->is_leaf) {
            if (candidate.ends_with(suffix)) { // need C++20
                ans = max(ans, node->val);
            }
        }
        for (char c='a'; c<='z'; ++c) {
            if (node->children[c] != nullptr) {
                continue;
            }
            // perform backtrace
            candidate.push_back(c);
            backtrace(node->children[c]);
            candidate.pop_back();
        }
    };
    backtrace(p);
    return ans;
}


void WordFilter_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    WordFilter tm;
    int n = funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "buildDict") {
            tm.buildDict(funcArgs[i]);
            SPDLOG_INFO("{}({}) passed", funcOperations[i], numberVectorToString<std::string>(funcArgs[i]));
        } else if (funcOperations[i] == "filter") {
            int actual = tm.filter(funcArgs[i][0], funcArgs[i][1]);
            if (actual == std::stoi(ans[i])) {
                SPDLOG_INFO("{}({}, {}, {}) passed", funcOperations[i], funcArgs[i][0], funcArgs[i][1], ans[i]);
            } else {
                SPDLOG_ERROR("{}({}, {}) failed. Expected={}, actual={}", funcOperations[i], funcArgs[i][0], funcArgs[i][1], ans[i], actual);
            }
        }
    }
}


int main() {
    SPDLOG_WARN("Running WordFilter tests:");
    TIMER_START(WordFilter);
    WordFilter_scaffold(
        "[WordFilter,buildDict,filter,filter,filter,filter,filter]", 
        "[[],[apple,abyss,hello,hero],[a,e],[b,,],[a,,],[h,o],[x,y]]",
        "[nullptr,nullptr,0,-1,1,3,-1]");
    TIMER_STOP(WordFilter);
    SPDLOG_WARN("WordFilter tests use {} ms", TIMER_MSEC(WordFilter));
}
