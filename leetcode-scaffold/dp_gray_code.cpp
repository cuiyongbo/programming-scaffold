#include "leetcode.h"

using namespace std;

/* leetcode: 89 */
class Solution {
public:
    vector<int> grayCode(int n);
};


/*
The gray code is a binary numeral system where two successive values differ in only one bit. A gray code sequence must begin with 0.
Given a non-negative integer n representing the total number of bits in the code, (the maximum would be `2^n - 1`) print the sequence of gray code.
Note: refer to <hacker's delight> ch13 for further info. 
*/
vector<int> Solution::grayCode(int n) {

{ // dp solution with optimization of space usage
    vector<int> ans; ans.reserve(1<<n);
    ans.push_back(0); // trivial case
    for (int i=1; i<=n; i++) {
        // ans is already a sequence where two adjacent values differ in only one bit
        int h = 1 << (i-1);
        int sz = ans.size();
        for (int j=sz-1; j>=0; j--) {
            // add one bit to each element in the reverse order of ans
            ans.push_back(h|ans[j]);
        }
    }
    return ans;
}

{ // naive dp solution
    // dp[i] means the graycode with i bits
    // dp[i] = dp[i-1] + {x|(1<<(i-1)) for x in reversed(dp[i-1])}
    // dp[0] = {0}
    vector<vector<int>> dp(n+1);
    dp[0] = {0}; // trivial cases
    for (int i=1; i<=n; ++i) {
        dp[i] = dp[i-1];
        // dp[i-1] is already a subsequence where two successive values differ in only one bit
        for (int j=dp[i-1].size()-1; j>=0; --j) {
            dp[i].push_back(dp[i-1][j] | (1<<(i-1)));
        }
    }
    return dp[n];
}

}


void grayCode_scaffold(int input, string expectedResult) {
    Solution ss;
    auto expected = stringTo1DArray<int>(expectedResult);
    auto actual = ss.grayCode(input);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, numberVectorToString<int>(actual));
    }
}


int main() {
    SPDLOG_WARN("Running grayCode tests:");
    TIMER_START(grayCode);
    grayCode_scaffold(0, "[0]");
    grayCode_scaffold(1, "[0,1]");
    grayCode_scaffold(2, "[0,1,3,2]");
    grayCode_scaffold(3, "[0,1,3,2,6,7,5,4]");
    grayCode_scaffold(4, "[0,1,3,2,6,7,5,4,12,13,15,14,10,11,9,8]");
    TIMER_STOP(grayCode);
    SPDLOG_WARN("grayCode tests use {} ms", TIMER_MSEC(grayCode));
}
