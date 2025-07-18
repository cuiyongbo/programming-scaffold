#include "leetcode.h"

using namespace std;

/*
leetcode: 1143 

Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
For example, "ace" is a subsequence of "abcde". A common subsequence of two strings is a subsequence that is common to both strings.

Example 1:
    Input: text1 = "abcde", text2 = "ace" 
    Output: 3  
    Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:
    Input: text1 = "abc", text2 = "def"
    Output: 0
    Explanation: There is no such common subsequence, so the result is 0.
*/

class Solution {
public:
    int lengthOflongestCommonSubsequence(string x, string y);
    string longestCommonSubsequence(string x, string y);
};

int Solution::lengthOflongestCommonSubsequence(string x, string y) {
{ // naive solution
    int m = x.size();
    int n = y.size();
    // dp[i][j] means the lengthOflongestCommonSubsequence(x[:i], y[:j]). i, j are not inclusive
    // dp[i][j] = dp[i-1][j-1]+1 if x[i]==y[j] else
    //      dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            // dp[i] is never used after we have computed dp[i+1]
            // and this is a hint to optimize the space usage of the algorithm
            if (x[i-1] == y[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}

{ // solution with opmization of space usage
    int m = x.size();
    int n = y.size();
    vector<vector<int>> dp(2, vector<int>(n+1, 0));
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            if (x[i-1] == y[j-1]) {
                dp[1][j] = dp[0][j-1] + 1;
            } else {
                dp[1][j] = max(dp[0][j], dp[1][j-1]);
            }
        }
        dp[0] = dp[1];
    }
    return dp[1][n];
}

}

string Solution::longestCommonSubsequence(string x, string y) {
    int m = x.size();
    int n = y.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            if (x[i-1] == y[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

if (1) { // iterative solution
    string candidate;
    int i = m;
    int j = n;
    while (i>0 && j>0) {
        if (x[i-1] == y[j-1]) {
            candidate.push_back(x[i-1]);
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else if (dp[i-1][j] < dp[i][j-1]) {
            j--;
        } else {
            // there are multiple candidates, either direction would be fine
            i--;
        }
    }
    std::reverse(candidate.begin(), candidate.end());
    return candidate;
}

{ // recursive version
    string candidate;
    set<string> ans; // Note that there may be duplicates
    function<void(int, int)> backtrace = [&] (int i, int j) {
        if (i==0 || j==0) { // terminate
            if (!candidate.empty()) { // omit invalid candidate(s)
                ans.emplace(candidate);
            }
            return;
        }
        if (x[i-1] == y[j-1]) {
            // perform backtrace here
            candidate.push_back(x[i-1]);
            backtrace(i-1, j-1);
            candidate.pop_back();
        } else if (dp[i-1][j] > dp[i][j-1]) {
            backtrace(i-1, j);
        } else if (dp[i-1][j] < dp[i][j-1]){
            backtrace(i, j-1);
        } else { // traverse both routes
            backtrace(i-1, j);
            backtrace(i, j-1);
        }
    };

    backtrace(m, n);
    
    if (ans.empty()) {
        return "";
    } else {
        string p = *(ans.begin());
        reverse(p.begin(), p.end());
        return p;
    }
}

}


void longestCommonSubsequence_scaffold(string x, string y, string lcs) {
    Solution ss;
    int actual_len = ss.lengthOflongestCommonSubsequence(x, y); 
    string actual_lcs = ss.longestCommonSubsequence(x, y); 
    if (actual_len == lcs.size() && (actual_lcs == lcs || actual_lcs.length() == lcs.length())) {
        SPDLOG_INFO("case({}, {}, lcs={}) passed", x, y, lcs);
    } else {
        SPDLOG_ERROR("case({}, {}, lcs={}) failed, actual: {}", x, y, lcs, actual_lcs);
    }
}


int main() {
    SPDLOG_WARN("Running longestCommonSubsequence tests:");
    TIMER_START(longestCommonSubsequence);
    longestCommonSubsequence_scaffold("xaxyyy", "xxaxbyy", "xaxyy");
    longestCommonSubsequence_scaffold("abcde", "ace", "ace");
    longestCommonSubsequence_scaffold("abc", "abc", "abc");
    longestCommonSubsequence_scaffold("abc", "def", "");
    longestCommonSubsequence_scaffold("abcba", "abcbcba", "abcba");
    longestCommonSubsequence_scaffold("bsbininm", "jmjkbkjkv", "m");
    TIMER_STOP(longestCommonSubsequence);
    SPDLOG_WARN("longestCommonSubsequence tests use {} ms", TIMER_MSEC(longestCommonSubsequence));
    return 0;
}
