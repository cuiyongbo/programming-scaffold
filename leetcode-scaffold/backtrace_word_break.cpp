#include "leetcode.h"

using namespace std;

/* leetcode: 139, 140 */
class Solution {
public:
    bool wordBreak_139(string s, vector<string>& wordDict);
    vector<string> wordBreak_140(string s, vector<string>& wordDict);
};


/*
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
determine if s can be segmented into a space-separated sequence of one or more dictionary words. 
You may assume the dictionary does not contain duplicate words.
*/
bool Solution::wordBreak_139(string input, vector<string>& wordDict) {
    // ans = OR(wordBreak(input[:i]) && wordBreak(input[i:])), 0<i<input.size()
    map<string, bool> sub_solution;
    sub_solution[""] = true;
    for (auto& p: wordDict) {
        sub_solution[p] = true;
    }
    function<bool(string)> dfs = [&] (string u) {
        if (sub_solution.count(u)) { // memoization
            return sub_solution[u];
        }
        sub_solution[u] = false;
        // u is not in wordDict
        // Note that we start from i=1
        for (int i=1; i<(int)u.size(); i++) {
            // split u at index i
            auto lu = u.substr(0, i);
            auto ru = u.substr(i);
            // and test if lu and ru can be segmented into a space-separated sequence of one more dictionary word
            if (dfs(lu) && dfs(ru)) {
                sub_solution[u] = true;
                return true;
            }
        }
        return false;
    };
    return dfs(input);
}


/*
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
add spaces in s to construct a sentence where each word is a valid dictionary word. 
You may assume the dictionary does not contain duplicate words. Return all such possible sentences.
*/
vector<string> Solution::wordBreak_140(string input, vector<string>& wordDict) {
    using string_1d_vec_t = vector<string>;
    using string_2d_vec_t = vector<string_1d_vec_t>;
    map<string, string_2d_vec_t> sub_solution; // substring, splits
    std::set<string> word_set (wordDict.begin(), wordDict.end());
    function<string_2d_vec_t(string)> dfs = [&] (string u) {
        if (sub_solution.count(u)) { // memoization
            return sub_solution[u];
        }
        string_2d_vec_t ans;
        if (word_set.count(u)) { // test if u is in word_set
            ans.push_back({u});
        }
        for (int i=1; i<(int)u.size(); i++) {
            // split u at index i
            // and test if lu and ru can be segmented into a space-separated sequence of one more dictionary word
            auto l = dfs(u.substr(0, i));
            auto r = dfs(u.substr(i));
            if (!l.empty() && !r.empty()) {
                for (const auto& li: l) {
                    for (const auto& ri: r) {
                        string_1d_vec_t one_vec;
                        one_vec.insert(one_vec.end(), li.begin(), li.end());
                        one_vec.insert(one_vec.end(), ri.begin(), ri.end());
                        ans.push_back(one_vec);
                    }
                }
            }
        }
        sub_solution[u] = ans;
        return ans;
    };
    string_2d_vec_t result = dfs(input);
    set<string> ans;
    for (const auto& one_vec: result) {
        string candidate;
        for (const auto& w: one_vec) {
            candidate += w;
            candidate.push_back(' ');
        }
        candidate.pop_back();
        ans.insert(candidate);
    }
    return vector<string>(ans.begin(), ans.end());
}


void wordBreak_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input2);
    bool actual = ss.wordBreak_139(input1, dict);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


void wordBreakII_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input2);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    auto actual = ss.wordBreak_140(input1, dict);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        for (const auto& s: actual)  {
            std::cout << "[" << s << "]" << std::endl;
        }
    }
}


int main() {
    SPDLOG_WARN("Running wordBreak_139 tests:");
    TIMER_START(wordBreak_139);
    wordBreak_scaffold("leetcode", "[leet,code]", true);
    wordBreak_scaffold("leetcode", "[leet,code,loser]", true);
    wordBreak_scaffold("googlebingbaidu", "[google,bing,baidu]", true);
    wordBreak_scaffold("googlebingbaidu360", "[google,bing,baidu]", false);
    TIMER_STOP(wordBreak_139);
    SPDLOG_WARN("wordBreak_139 tests use {} ms", TIMER_MSEC(wordBreak_139));

    SPDLOG_WARN("Running wordBreak_140 tests:");
    TIMER_START(wordBreak_140);
    wordBreakII_scaffold("leetcode", "[leet,code]", "[leet code]");
    wordBreakII_scaffold("catsanddog", "[cat,cats,and,sand,dog]", "[cat sand dog,cats and dog]");
    TIMER_STOP(wordBreak_140);
    SPDLOG_WARN("wordBreak_140 tests use {} ms", TIMER_MSEC(wordBreak_140));
}
