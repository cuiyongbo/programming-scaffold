#include "leetcode.h"

using namespace std;

/*
Given a rod of length n inches and a table of prices :math:`p_i \text{ for i } \in [1,n]`, 
determine the maximum revenue :math:`r_n` obtainable by cutting up the rod and selling the pieces.
*/

class Solution {
public:
    int cut_rod(vector<int>& prices, int rod_len);
    // figure out the cut plan for an optimal revenue solution
    pair<int, string> extend_cut_rod(vector<int>& prices, int rod_len);
};

int Solution::cut_rod(vector<int>& prices, int rod_len) {
    int price_sz = prices.size();
    // rod_len is inclusive
    // revenue[i] means the maximum revenue obtained by cutting up a rod with length i
    // revenue[i] = max{price[j]+revenue[i-j]}, 0<j<=i
    vector<int> revenue(rod_len+1, INT32_MIN);
    revenue[0] = 0; // trivial case
    for (int i=1; i<=rod_len; ++i) {
        // i: length of rod
        int q = INT32_MIN;
        // j: how much rod to cut each time
        // Question: what if len(prices) is less than rod_len? then we have to cut rod into pieces which we may sell with prices
        for (int j=1; j<=min(i, price_sz-1); ++j) {
            q = max(q, prices[j]+revenue[i-j]);
        }
        revenue[i] = q;
    }
    // the answer
    return revenue[rod_len];
}

pair<int, string> Solution::extend_cut_rod(vector<int>& prices, int rod_len) {
    int price_sz = prices.size();
    // rod_len is inclusive
    vector<int> plan(rod_len+1, 0);
    vector<int> revenue(rod_len+1, INT32_MIN);
    revenue[0] = 0; // trivial case
    for (int i=1; i<=rod_len; ++i) {
        // i: length of rod
        int q = INT32_MIN;
        // j: how much rod to cut each time
        // Question: what if len(prices) is less than rod_len?
        for (int j=1; j<=min(i, price_sz-1); ++j) {
            if (q < prices[j]+revenue[i-j]) {
                q = prices[j]+revenue[i-j];
                plan[i] = j; // how much rod to cut each time
            }
        }
        // revenue[i] means the maximum revenue obtained by cutting up a rod with length i
        revenue[i] = q;
    }

    string ans;
    int n = rod_len;
    while (n>0) {
        ans.append(std::to_string(plan[n]));
        ans.append(",");
        n -= plan[n];
    }
    if (!ans.empty()) {
        ans.pop_back();
    } else {
        ans = "nil";
    }
    SPDLOG_INFO("case(prices={}, rod_len={}), max revenue: {}, solution: {}", numberVectorToString(prices), rod_len, revenue[rod_len], ans); 
    return {revenue[rod_len], ans};
}


void basic_test() {
    // prices[i] means the price of rod with length i is prices[i]
    vector<int> prices {
        0,1,5,8,9,10,17,17,20 // 0-8
    };

    Solution ss;
    for (int i=0; i<(int)prices.size(); ++i) {
        ss.extend_cut_rod(prices, i);
    }
}


int main() {
    basic_test();
}