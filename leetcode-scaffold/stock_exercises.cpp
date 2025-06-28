#include "leetcode.h"

using namespace std;

class Solution {
public:
    int maxProfit_121(vector<int>& prices);
    int maxProfit_122(vector<int>& prices);
    int maxProfit_123(vector<int>& prices);
    int maxProfit_188(int k, vector<int>& prices);
    int maxProfit_309(vector<int>& prices);
};


/*
Say you have an array for which the i-th element is the price of a given stock on day i.
If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.
*/
int Solution::maxProfit_121(vector<int>& prices) {
// dp[i] means maxProfit_121 when selling stock no later than i-th day
// dp[i] = max(dp[i-1], prices[i]-buy), buy = min(prices[k]), 0<=k<i
if (0) {
    int n = prices.size();
    vector<int> dp(n, 0);
    int buy = prices[0]; // initialization: we buy at the first day
    for (int i=1; i<n; i++) {
        dp[i] = max(dp[i-1], prices[i]-buy); // if we sell at day i, we can earn prices[i]-buy
        buy = min(buy, prices[i]); // choose a lower price to buy
    }
    return dp[n-1];
}

{ // solution with optimization of space usage
    int ans = 0;
    int buy = prices[0];
    int n = prices.size();
    for (int i=1; i<n; ++i) {
        ans = max(ans, prices[i]-buy);
        buy = min(buy, prices[i]);
    }
    return ans;
}

}


/*
Follow up:
You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
*/
int Solution::maxProfit_122(vector<int>& prices) {
    if (prices.empty()) {
        return 0;
    }
    int ans = 0;
    for (int i=1; i<(int)prices.size(); i++) {
        ans += max(0, // we don't perform transaction if we cann't make profit
             prices[i]-prices[i-1]);
    }
    return ans;
}


/*
Follow up: You may complete at most two transactions.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
*/
int Solution::maxProfit_123(vector<int>& prices) {
    if (prices.empty()) {
        return 0;
    }
    int f1 = -prices[0]; // maxProfit after the first purchase of the stock
    int f2 = 0;          // maxProfit after the first sale of the stock
    int f3 = -prices[0]; // maxProfit after the second purchase of the stock
    int f4 = 0;          // maxProfit after the second sale of the stock
    for (int i=1; i<(int)prices.size(); i++) {
        int p = prices[i];
        // We consider that buying and selling on the same day will result in a profit of 0, which will not affect the answer
        f1 = max(f1, -p);  // find a lower price to buy
        f2 = max(f2, f1+p); // find a higher price to sell
        f3 = max(f3, f2-p);
        f4 = max(f4, f3+p);
    }
    return f4;
}

/*
Follow up: You may complete at most k transactions.
*/
int Solution::maxProfit_188(int k, vector<int>& prices) {
    if (prices.empty()) {
        return 0;
    }
    int ans = 0;
    int n = prices.size();
    if(n > 2*k) {
        vector<int> global(k+1, 0), local(k+1, 0);
        for (int i=1; i<n; ++i) {
            int profit = prices[i] - prices[i-1]; // profit if we buy at day i-1 and sell at day i
            for (int j=k; j>0; --j) {
                local[j] = max(global[j-1] + max(0, profit), local[j]+profit);
                global[j] = max(global[j], local[j]);
            }
        }
        ans = global[k];
    } else {
        // we can perform multiple transactions to cover all stocks, so we do it when we can make profit from it
        // revert to leetcode 122
        for (int i=1; i<n; ++i) {
            ans += max(0, prices[i]-prices[i-1]);
        }
    }
    return ans;
}


/*
Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete as many transactions 
as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
    You may not engage in multiple transactions at the same day (i.e., you must sell the stock before you buy again).
    After you sell your stock, you cannot buy stock on next day. (i.e., cooldown 1 day)
*/
int Solution::maxProfit_309(vector<int>& prices) {

{ // naive solution
    if (prices.empty()) {
        return 0;
    }
    int n = prices.size();
    // Initialize the DP arrays
    vector<int> buy(n, 0), sell(n, 0), cooldown(n, 0);
    // buy[i] means maxProfit if you buy the stock on day i
    // sell[i] means maxProfit if you sell the stock on day i
    // cooldown[i] means maxProfit if you are in a cooldown period on day i (you sell the stock the day before or haven't done any transaction)
    // trivial cases:
    buy[0] = -prices[0]; // We bought a stock on the first day
    sell[0] = 0;          // Cannot sell on the first day without buying
    cooldown[0] = 0;      // No cooldown on the first day either
    // state transitions
    for (int i=1; i<n; ++i) {
        // the operation order doesn't matter
        // you can buy the stock on day i if you were in a cooldown period or you were already holding the stock
        buy[i] = max(buy[i-1], cooldown[i-1] - prices[i]);
        // you can sell the stock on day i if you were holding the stock the day before
        sell[i] = buy[i-1] + prices[i];
        // you can be in a cooldown period on day i if you were in a cooldown period or you just sell the stock the day before
        cooldown[i] = max(cooldown[i-1], sell[i-1]);
    }
    // The result is the maximum profit on the last day being in sell or cooldown states
    return max(sell[n-1], cooldown[n-1]);
}

{ // optimize space usage
    // sell[i] means maxProfit when sell 
    // buy[i] = max(buy[i-1], rest[i-1] - prices[i])
    // sell[i] = buy[i-1] + prices[i]
    // rest[i] = max(rest[i-1], sell[i-1])
    // init: rest[0]=sell[0]=0, buy[0]=-inf
    int sell = 0;
    int rest = 0;
    int buy = -prices[0];
    for (auto p: prices) {
        int ps=sell, pr=rest, ph=buy;
        sell = ph + p;
        buy = max(ph, pr-p);
        rest = max(pr, ps);
    }
    return max(sell, rest);
}

}


void maxProfit_scaffold(string input, int expectedResult, int func_no) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = 0;
    if (func_no == 121) {
        actual = ss.maxProfit_121(prices);
    } else if (func_no == 309) {
        actual = ss.maxProfit_309(prices);
    } else if (func_no == 122) {
        actual = ss.maxProfit_122(prices);
    } else if (func_no == 123) {
        actual = ss.maxProfit_123(prices);
    } else {
        SPDLOG_ERROR("func_no must be one in [121, 309, 122], actual: {}", func_no);
        return;
    }
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}, func_no={}) passed", input, expectedResult, func_no);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}, func_no={}) failed, actual: {}", input, expectedResult, func_no, actual);
    }
}


void maxProfit_188_scaffold(string input, int k, int expectedResult) {
    vector<int> prices = stringTo1DArray<int>(input);
    Solution ss;
    int actual = ss.maxProfit_188(k, prices);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, k, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, k, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running maxProfit tests:");
    TIMER_START(maxProfit);
    maxProfit_scaffold("[7, 1, 5, 3, 6, 4]", 5, 121);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0, 121);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0, 122);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0, 123);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0, 309);
    maxProfit_scaffold("[1, 2, 3, 0, 2]", 3, 309);
    maxProfit_scaffold("[1]", 0, 309);
    maxProfit_scaffold("[1]", 0, 121);
    maxProfit_scaffold("[7,1,5,3,6,4]", 7, 122);
    maxProfit_scaffold("[1,2,3,4,5]", 4, 122);
    maxProfit_scaffold("[1,2,3,4,5]", 4, 123);
    maxProfit_scaffold("[3,3,5,0,0,3,1,4]", 6, 123);
    maxProfit_scaffold("[3,3,5,0,0,3,1,4]", 6, 123);
    TIMER_STOP(maxProfit);
    SPDLOG_WARN("maxProfit tests use {} ms", TIMER_MSEC(maxProfit));

    SPDLOG_WARN("Running maxProfit_188 tests:");
    TIMER_START(maxProfit_188);
    maxProfit_188_scaffold("[7,1,5,3,6,4]", 1, 5);
    maxProfit_188_scaffold("[7,6,4,3,1]", 1, 0);
    maxProfit_188_scaffold("[7,6,4,3,1]", 2, 0);
    maxProfit_188_scaffold("[7,6,4,3,1]", 3, 0);
    maxProfit_188_scaffold("[7,1,5,3,6,4]", 2, 7);
    maxProfit_188_scaffold("[1,2,3,4,5]", 1, 4);
    maxProfit_188_scaffold("[1,2,3,4,5]", 2, 4);
    maxProfit_188_scaffold("[2,4,1]", 2, 2);
    maxProfit_188_scaffold("[3,2,6,5,0,3]", 2, 7);
    TIMER_STOP(maxProfit_188);
    SPDLOG_WARN("maxProfit_188 tests use {} ms", TIMER_MSEC(maxProfit_188));
}
