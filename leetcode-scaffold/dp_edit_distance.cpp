#include "leetcode.h"

using namespace std;

/* leetcode: 10, 72 */
class Solution {
public:
    bool isMatch(string s, string p);
    int minDistance(string word1, string word2);
    bool isInterleave(string s1, string s2, string s3);

private:
    int minDistance_dp(string word1, string word2);
    int minDistance_memoization(string word1, string word2);
};

/*
Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)
You have the following 3 operations permitted on a word:
    a) Insert a character
    b) Delete a character
    c) Replace a character
*/
int Solution::minDistance(string word1, string word2) {
    // return minDistance_dp(word1, word2);
    return minDistance_memoization(word1, word2);
}

int Solution::minDistance_dp(string word1, string word2) {
    // dp[i][j] means minDistance(word1[0,i), word2[0, j)). i, j are not inclusive
    // dp[i][j] = dp[i-1][j-1] if word1[i-1]=word2[j-1] else 
    //      dp[i][j] = min{dp[i][j-1], dp[i-1][j], dp[i-1][j-1]} + 1
    int m = word1.size();
    int n = word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, INT32_MAX));
    // trivial cases:
    for (int i=0; i<=n; i++) {
        dp[0][i] = i; // insertion. word1 is empty
    }
    for (int i=0; i<=m; i++) {
        dp[i][0] = i; // deletion. word2 is empty
    }
    // recursion: 
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1]; // no operation
            } else {
                dp[i][j] = min(dp[i][j], dp[i-1][j-1]+1); // replacement
                dp[i][j] = min(dp[i][j], dp[i][j-1]+1); // insertion. we insert word2[j] to word1, then we consider how to convert word1[:i] to word2[:j-1]
                dp[i][j] = min(dp[i][j], dp[i-1][j]+1); // deletion. we delete word1[i] from word1, then we consider how to convert word1[:i-1] to word2[:j]
            }
        }
    }
    return dp[m][n];
}


int Solution::minDistance_memoization(string word1, string word2) {
    int m = word1.size();
    int n = word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, -1));
    function<int(int, int)> dfs = [&] (int l1, int l2) {
        if (l1 == 0) {
            return l2;
        }
        if (l2 == 0) {
            return l1;
        }
        if (dp[l1][l2] >= 0) {
            return dp[l1][l2];
        }
        if (word1[l1-1] == word2[l2-1]) {
            dp[l1][l2] = dfs(l1-1, l2-1);
        } else {
            dp[l1][l2] = 1 + min(dfs(l1-1, l2-1), min(dfs(l1-1, l2), dfs(l1, l2-1)));
        }
        return dp[l1][l2];
    };
    return dfs(m, n);
}


/*
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
'.' Matches any single character. and '*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
Note:
    s could be empty or contains only lowercase letters a-z.
    p could be empty or contains only lowercase letters a-z, and characters like . or  *.
For example, given an input: s = "ab", p = ".*", output: true, explanation: ".*" means "zero or more (*) of any character (.)".
*/
bool Solution::isMatch(string s, string p) {
    int m = s.size(), n = p.size();
    // dp[i][j] means whether p[:j] matches s[:i] or not (**the right ends are not inclusive**)
    vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
    // initialization:
    dp[0][0] = true; // both s, p are empty
    for (int j=2; j<=n; ++j) {
        if (p[j-1] == '*') { // trivial cases: p matches an empty string
            dp[0][j] = dp[0][j-2];
        }
    }
    // i, j are not inclusive
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            if (p[j-1] == '.') { // a "." matches all possible characters, so p[j-1] matches s[i-1]
                dp[i][j] = dp[i-1][j-1];
            } else if (p[j-1] == '*') {
                if (j>1) {
                    // 1. Use '*' to match zero character in s
                    dp[i][j] = dp[i][j-2]; // why `j-2`? because '*' matches zero or more of *the preceding element.* and p[j-1] is omitted in this case
                    // 2. Use '*' to match one or more characters in s. p[j-1] is used as p[j-2]
                    if (s[i-1] == p[j-2] /*exact match*/
                        || p[j-2]=='.' /*wildcard*/) {
                        dp[i][j] = dp[i][j] || dp[i-1][j];
                    }
                } else {
                    // no preceding element in p
                }
            } else { // cases require an exact match
                dp[i][j] = (s[i-1] == p[j-1]) ? dp[i-1][j-1] : false;
            }
        }
    }
    return dp[m][n];
}


void minDistance_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    int actual = ss.minDistance(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


void isMatch_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    bool actual = ss.isMatch(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
An interleaving of two strings s and t is a configuration where s and t are divided into n and m substrings respectively, such that:

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
Note: a + b is the concatenation of strings a and b.

Example 1:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
Since s3 can be obtained by interleaving s1 and s2, we return true.

Example 2:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.

Example 3:
Input: s1 = "", s2 = "", s3 = ""
Output: true
 
Constraints:
0 <= s1.length, s2.length <= 100
0 <= s3.length <= 200
s1, s2, and s3 consist of lowercase English letters.

Follow up: Could you solve it using only O(s2.length) additional memory space?
*/
bool Solution::isInterleave(string s1, string s2, string s3) {
    int l1 = s1.size();
    int l2 = s2.size();
    int l3 = s3.size();
    if (l1+l2 != l3) {
        return false;
    }
    // dp[i][j] means whether s3[0:i+j] can be formed by interleaving by s1[0:i] and s2[0:j]
    vector<vector<int>> dp(l1+1, vector<int>(l2+1, false));
    dp[0][0] = true; // initialization
    for (int i=0; i<=l1; i++) {
        for (int j=0; j<=l2; j++) {
            // i, j are not inclusive
            if (i > 0) {
                dp[i][j] |= dp[i-1][j] && s1[i-1]==s3[i+j-1]; // s3[i+j-1] = s2[0:j] + s1[0:i] 
            }
            if (j > 0) {
                dp[i][j] |= dp[i][j-1] && s2[j-1]==s3[i+j-1]; // s3[i+j-1] = s1[0:i] + s2[0:j]
            }
        }
    }
    return dp[l1][l2];
}


void isInterleave_scaffold(string input1, string input2, string input3, int expectedResult) {
    Solution ss;
    bool actual = ss.isInterleave(input1, input2, input3);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual: {}", input1, input2, input3, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running minDistance tests:");
    TIMER_START(minDistance);
    minDistance_scaffold("hello", "hope", 4);
    minDistance_scaffold("horse", "ros", 3);
    minDistance_scaffold("intention", "execution", 5);
    TIMER_STOP(minDistance);
    SPDLOG_WARN("minDistance tests use {} ms", TIMER_MSEC(minDistance));

    SPDLOG_WARN("Running isMatch tests:");
    TIMER_START(isMatch);
    isMatch_scaffold("aa", "a", false);
    isMatch_scaffold("aa", "a*", true);
    isMatch_scaffold("ab", ".*", true);
    isMatch_scaffold("aab", "c*a*b*", true);
    isMatch_scaffold("mississippi", "mis*is*p*.", false);
    isMatch_scaffold("aaa", "ab*ac*a", true);
    TIMER_STOP(isMatch);
    SPDLOG_WARN("isMatch tests use {} ms", TIMER_MSEC(isMatch));

    SPDLOG_WARN("Running isInterleave tests:");
    TIMER_START(isInterleave);
    isInterleave_scaffold("aabcc", "dbbca", "aadbbcbcac", 1);
    isInterleave_scaffold("aabcc", "dbbca", "aadbbbaccc", 0);
    isInterleave_scaffold("", "", "", 1);
    TIMER_STOP(isInterleave);
    SPDLOG_WARN("isInterleave tests use {} ms", TIMER_MSEC(isInterleave));
}
