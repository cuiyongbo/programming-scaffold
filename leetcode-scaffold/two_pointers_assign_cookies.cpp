#include "leetcode.h"

using namespace std;

/* leetcode: 455, 209 */
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s);
    int minSubArrayLen(int s, vector<int>& nums);
};

/*
Assume you are an awesome parent and want to give your children some cookies. 
But you should give each child at most one cookie. Each child i has a greed factor g_i, 
which is the minimum size of a cookie that the child will be content with; 
and each cookie j has a size s_j. If s_j >= g_i, we can assign the cookie j to the child i, 
and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.

Note:
    You may assume the greed factor is always positive.
    You cannot assign more than one cookie to one child.
*/
int Solution::findContentChildren(vector<int>& g, vector<int>& s) {
    // sort the arrays in ascending order
    sort(g.begin(), g.end(), std::less<int>());
    sort(s.begin(), s.end(), std::less<int>());
    int child_cnt = g.size();
    int cookie_cnt = s.size();
    int child = 0;
    int cookie = 0;
    while (cookie<cookie_cnt && child<child_cnt) {
        if (g[child]>s[cookie]) { // find the smallest cookie that satisfy current child
            cookie++;
        } else {
            child++;
            cookie++;
        }
    }
    return child;
}


/*
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray whose sum ≥ s. 
If there isn’t one, return 0 instead.
*/
int Solution::minSubArrayLen(int s, vector<int>& nums) {
    int ans = INT32_MAX;
    int sum = 0;
    int sz = nums.size();
    for (int i=0, j=0; i<sz; i++) {
        while (j<sz&&sum<s) {
            sum += nums[j];
            j++; // j is not inclusive when calculating sum
        }
        if (sum < s) {
            break;
        }
        assert(sum>=s);
        //printf("ans=%d, j=%d, i=%d\n", ans, j, i);
        ans = min(ans, j-i); // why not j-i+1, because j is not inclusive when calculating sum
        sum -= nums[i]; // excluse nums[i] from next iteration, but we can reuse sum{nums[i+1:j]}
    }
    return ans==INT32_MAX ? 0 : ans;
}


void findContentChildren_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input1);
    vector<int> s = stringTo1DArray<int>(input2);
    int actual = ss.findContentChildren(g, s);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


void minSubArrayLen_scaffold(int input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input2);
    int actual = ss.minSubArrayLen(input1, nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running findContentChildren tests:");
    TIMER_START(findContentChildren);
    findContentChildren_scaffold("[1,2,3]", "[1,1]", 1);
    findContentChildren_scaffold("[1,2]", "[1,2,3]", 2);
    TIMER_STOP(findContentChildren);
    SPDLOG_WARN("findContentChildren tests use {} ms", TIMER_MSEC(findContentChildren));

    SPDLOG_WARN("Running minSubArrayLen tests:");
    TIMER_START(minSubArrayLen);
    minSubArrayLen_scaffold(7, "[2,3,1,2,4,3]", 2);
    minSubArrayLen_scaffold(6, "[2,3,1,2,4,3]", 2);
    minSubArrayLen_scaffold(4, "[2,3,1,2,4,3]", 1);
    minSubArrayLen_scaffold(9, "[2,3,1,2,4,3]", 3);
    TIMER_STOP(minSubArrayLen);
    SPDLOG_WARN("minSubArrayLen tests use {} ms", TIMER_MSEC(minSubArrayLen));
}