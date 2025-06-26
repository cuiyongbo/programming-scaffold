#include "leetcode.h"

using namespace std;

// https://www.geeksforgeeks.org/dsa/0-1-knapsack-problem-dp-10/#tabulation-or-bottomup-approach-on-x-w-time-and-space
/*
Given n items where each item has some weight and profit associated with it and also given a bag with capacity W, [i.e., the bag can hold at most W weight in it]. The task is to put the items into the bag such that the sum of profits associated with them is the maximum possible. 

Note: The constraint here is we can either put an item completely into the bag or cannot put it at all [It is not possible to put a part of an item into the bag].

Input:  W = 4, profit = [1, 2, 3], weight = [4, 5, 1]
Output: 3
Explanation: There are two items which have weight less than or equal to 4. If we select the item with weight 4, the possible profit is 1. And if we select the item with weight 1, the possible profit is 3. So the maximum possible profit is 3. Note that we cannot put both the items with weight 4 and 1 together as the capacity of the bag is 4.

Input: W = 3, profit = [1, 2, 3], weight = [4, 5, 6]
Output: 0

Similar problems:
 - 518. Coin Change II
*/

class Solution {
public:
    int knapsack(int W, vector<int>& profit, vector<int>& weight);
    int coinChange(vector<int>& coins, int amount);
    int combinationSum_322(vector<int>& nums, int target);
    int findTargetSumWays(vector<int>& nums, int target);
};


int Solution::knapsack(int W, vector<int>& profit, vector<int>& weight) {
if (0) { // backtrace solution
    int n = profit.size();
    vector<bool> used(n, false);
    // return the max profit we may get with bag capcity of cap
    function<int(int)> backtrace = [&] (int cap) {
        if (cap <= 0) { // termination
            return 0;
        }
        int ans = 0;
        for (int i=0; i<n; i++) {
            if (used[i]) {
                continue;
            }
            if (cap<weight[i]) { // prune invalid branches
                continue;
            }
            used[i] = true;
            ans = max(ans, backtrace(cap-weight[i])+profit[i]);
            used[i] = false;
        }
        return ans;
    };
    return backtrace(W);
}

{ // dp solution
    int n = profit.size();
    vector<vector<int>> dp(n+1, vector<int>(W+1, 0));
    // dp[i][j] means the max profit we can get with weight[:i] package and a bag of weight j
    // dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i-1]])
    for (int i=1; i<=n; i++) {
        for (int j=1; j<=W; j++) {
            int not_pick_package_i = dp[i-1][j];
            int pick_package_i = 0;
            if (j>=weight[i-1]) {
                pick_package_i = profit[i-1] + dp[i-1][j-weight[i-1]];
            }
            dp[i][j] = max(not_pick_package_i, pick_package_i);
        }
    }
    return dp[n][W];
}

}


/*
You are given coins of different denominations and a total amount of money amount.
Write a function to compute the fewest number of coins that you need to make up that amount.
If that amount of money cannot be made up by any combination of the coins, return -1.
Example 1:
    coins = [1, 2, 5], amount = 11
    return 3 (11 = 5 + 5 + 1)
*/
int Solution::coinChange(vector<int>& coins, int amount) {
    // dp[i] means the minimum number of coins to make up amount i
    // dp[i] = min{dp[i-coin]+1} for coin in coins
    // NOTE that we use INT32_MAX as invalid value, NOT -1
    vector<int> dp(amount+1, INT32_MAX);
    dp[0] = 0; // initialization
    for (int coin: coins) {
        for (int i=coin; i<=amount; ++i) {
            if (dp[i-coin] != INT32_MAX) {
                dp[i] = min(dp[i], dp[i-coin]+1);
            }
        }
    }
    return dp[amount]==INT32_MAX ? -1 : dp[amount];
}


/*
Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.
Given inputs: nums = [1, 2, 3], target = 4, The possible combination ways are:
    (1, 1, 1, 1)
    (1, 1, 2)
    (1, 2, 1)
    (1, 3)
    (2, 1, 1)
    (2, 2)
    (3, 1)
Note that different sequences are counted as different combinations. Therefore the output is 7.
I don't think this is combination
*/
int Solution::combinationSum_322(vector<int>& nums, int target) {
    // dp[i] means the number of combination in nums whose sum up to i
    // dp[i] = sum(dp[i-n]) for n in nums if (i-n)>=0
    vector<int> dp(target+1, 0);
    dp[0] = 1; // for case n==target
    for (int i=1; i<=target; ++i) {
        for (int n: nums) {
            if (i-n >= 0) {
                dp[i] += dp[i-n];
            }
        }
    }
    return dp[target];
}


/*
You are given an integer array nums and an integer target. You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.

Example 1:
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

Example 2:
Input: nums = [1], target = 1
Output: 1
 
Constraints:
1 <= nums.length <= 20
0 <= nums[i] <= 1000
0 <= sum(nums[i]) <= 1000
-1000 <= target <= 1000

Similar problems:
    - addOperators
*/
int Solution::findTargetSumWays(vector<int>& nums, int target) {
if (0) { // backtrace solution
    int sz = nums.size();
    vector<int> ops(sz, 0);
    auto is_valid = [&] () {
        int sum = 0;
        for (int i=0; i<sz; i++) {
            sum += ops[i] * nums[i];
        }
        return sum == target;
    };
    function<int(int)> backtrace = [&] (int u) {
        if (u == sz) {
            return is_valid() ? 1 : 0;
        }
        int ans = 0;
        {
            ops[u] = 1;
            ans += backtrace(u+1);
            ops[u] = 0;
        }
        {
            ops[u] = -1;
            ans += backtrace(u+1);
            ops[u] = 0;
        }
        return ans;
    };
    return backtrace(0);
}

if (0) { // dp solution
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    if (sum < std::abs(target)) { // trivial case
        return 0;
    }
    int offset = sum;
    int max_n = 2*sum+1;
    int sz = nums.size();
    vector<vector<int>> dp(sz+1, vector<int>(max_n, 0));
    // dp[i][j] the number of ways to build an expression with nums[:i] which is evaluated to (j-offset)
    dp[0][offset] = 1; // initialization, target == 0
    for (int i=1; i<=sz; i++) {
        for (int j=0; j<max_n; j++) {
            dp[i][j] += dp[i-1][j-nums[i-1]]; // add '+'
            dp[i][j] += dp[i-1][j+nums[i-1]]; // add '-'
        }
    }
    return dp[sz][target+offset]; // map target to [0, 2*sum+1]
}

{ // refined dp solution
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    if (sum < std::abs(target)) { // trivial case
        return 0;
    }
    int offset = sum;
    int max_n = 2*sum+1;
    vector<int> ways(max_n, 0); // map [-sum, sum] to [0, 2*sum+1]
    ways[offset] = 1; // initialization, target == 0
    for (auto n: nums) {
        vector<int> tmp(max_n, 0);
        for (int i=0; i<max_n; i++) {
            if (i+n < max_n) {
                tmp[i] += ways[i+n];
            }
            if (i-n >= 0) {
                tmp[i] += ways[i-n];
            }
        }
        std::swap(tmp, ways);
    }
    return ways[target+offset]; // map target to [0, 2*sum+1]
}

}


void knapsack_scaffold(int input1, string input2, string input3, int expectedResult) {
    Solution ss;
    vector<int> profit = stringTo1DArray<int>(input2);
    vector<int> weight = stringTo1DArray<int>(input3);
    int actual = ss.knapsack(input1, profit, weight);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual: {}", input1, input2, input3, expectedResult, actual);
    }
}


void coinChange_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input1);
    int actual = ss.coinChange(vi, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


void combinationSum_322_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input1);
    int actual = ss.combinationSum_322(vi, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


void findTargetSumWays_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input1);
    int actual = ss.findTargetSumWays(vi, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running knapsack tests:");
    TIMER_START(knapsack);
    knapsack_scaffold(3, "[1,2,3]", "[4,5,6]", 0);
    knapsack_scaffold(3, "[1,2,3]", "[4,5,1]", 3);
    TIMER_STOP(knapsack);
    SPDLOG_WARN("knapsack tests use {} ms", TIMER_MSEC(knapsack));

    SPDLOG_WARN("Running coinChange tests:");
    TIMER_START(coinChange);
    coinChange_scaffold("[1,2,5]", 11, 3);
    coinChange_scaffold("[2]", 3, -1);
    TIMER_STOP(coinChange);
    SPDLOG_WARN("coinChange tests use {} ms", TIMER_MSEC(coinChange));

    SPDLOG_WARN("Running combinationSum_322 tests:");
    TIMER_START(combinationSum_322);
    combinationSum_322_scaffold("[1,2,3]", 4, 7);
    combinationSum_322_scaffold("[1,2,3,4]", 4, 8);
    combinationSum_322_scaffold("[9]", 4, 0);
    TIMER_STOP(combinationSum_322);
    SPDLOG_WARN("combinationSum_322 tests use {} ms", TIMER_MSEC(combinationSum_322));

    SPDLOG_WARN("Running findTargetSumWays tests:");
    TIMER_START(findTargetSumWays);
    findTargetSumWays_scaffold("[1,1,1,1,1]", 3, 5);
    findTargetSumWays_scaffold("[1]", 1, 1);
    TIMER_STOP(findTargetSumWays);
    SPDLOG_WARN("findTargetSumWays tests use {} ms", TIMER_MSEC(findTargetSumWays));
}
