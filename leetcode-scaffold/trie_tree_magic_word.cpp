#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;

/*
leetcode: 676
Implement a magic directory with buildDict, and search methods.
For the method buildDict, you’ll be given a list of non-repetitive words to build a dictionary.
For the method search, you’ll be given a word, and judge whether if you modify exactly one character 
into another character in this word, the modified word is in the dictionary you just built.
Note: You may assume that all the inputs are consist of lowercase letters a-z.

Example 1:
    Input: buildDict(["hello", "leetcode"]), Output: Null
    Input: search("hello"), Output: False
    Input: search("hhllo"), Output: True
    Input: search("hell"), Output: False
    Input: search("leetcoded"), Output: False
*/
class MagicDictionary {
public:
    void buildDict(const vector<string>& dict);
    bool search(string word);
private:
    TrieTree m_trie_tree;
};

void MagicDictionary::buildDict(const vector<string>& dict) {
    for (const auto& s: dict) {
        m_trie_tree.insert(s);
    }
}

bool MagicDictionary::search(string word) {
    for (int i=0; i<(int)word.size(); ++i) {
        for (char c='a'; c<='z'; ++c) { // at each level, we may change one character into another, then test if the changed word exists in the dict
            if (word[i] == c) { // skip oneself
                continue;
            }
            char prev = word[i];
            word[i] = c;
            if (m_trie_tree.search(word)) { // search the modified word from scratch
                return true;
            }
            word[i] = prev; // restore
        }
    }   
    return false;
}


void MagicDictionary_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    MagicDictionary tm;
    int n = funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "buildDict") {
            tm.buildDict(funcArgs[i]);
            SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
        } else if (funcOperations[i] == "search") {
            bool actual = tm.search(funcArgs[i][0]);
            string actual_str = actual ? "true" : "false";
            if (actual_str == ans[i]) {
                SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
            } else {
                SPDLOG_ERROR("{}({}) failed. ExpectedResult={}, actual={}", funcOperations[i], funcArgs[i][0], ans[i], actual);
            }
        }
    }
}


int main() {
    SPDLOG_WARN("Running MagicDictionary tests:");
    TIMER_START(MagicDictionary);
    MagicDictionary_scaffold(
        "[MagicDictionary,buildDict,search,search,search,search,search,search]", 
        "[[],[hello,leetcode,hero],[hello],[hhllo],[hell],[leetcoded],[hellp],[pello]]",
        "[null,null,false,true,false,false,true,true]");
    TIMER_STOP(MagicDictionary);
    SPDLOG_WARN("MagicDictionary tests use {} ms", TIMER_MSEC(MagicDictionary));
}
