#include "leetcode.h"

using namespace std;

/* leetcode: 455, 209, 560 */
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s);
    int numRescueBoats(vector<int>& people, int limit);

};


/*
The i-th person has weight people[i], and each boat can carry a maximum weight of `limit`.
Each boat carries at most 2 people at the same time, provided the maximum weight of those people is at most `limit`. so It is guaranteed each person can be carried by a boat.
Return the minimum number of boats to carry every given person.

Example 2:
Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)

Example 3:
Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)
*/
int Solution::numRescueBoats(vector<int>& people, int limit) {
    // sort the weights in descending order since we need carry people with more weight first
    std::sort(people.begin(), people.end(), std::greater<int>());
    int ans = 0;
    int l = 0;
    int r = people.size() - 1; // r is inclusive
    while (l <= r) {
        // can we take another people with less weight?
        if (l != r && people[l]+people[r]<=limit) { // can we take two people in one boat?
            --r;
        }
        ++ans; ++l;
    }
    return ans;
} 


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


void numRescueBoats_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    int actual = ss.numRescueBoats(A, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
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


int main() {
    SPDLOG_WARN("Running numRescueBoats tests:");
    TIMER_START(numRescueBoats);
    numRescueBoats_scaffold("[1,2]", 3, 1 );
    numRescueBoats_scaffold("[3,2,2,1]", 3, 3);
    numRescueBoats_scaffold("[3,5,3,4]", 5, 4);
    numRescueBoats_scaffold("[3,1,3,1,3]", 4, 3);
    TIMER_STOP(numRescueBoats);
    SPDLOG_WARN("numRescueBoats tests use {} ms", TIMER_MSEC(numRescueBoats));

    SPDLOG_WARN("Running findContentChildren tests:");
    TIMER_START(findContentChildren);
    findContentChildren_scaffold("[1,2,3]", "[1,1]", 1);
    findContentChildren_scaffold("[1,2]", "[1,2,3]", 2);
    TIMER_STOP(findContentChildren);
    SPDLOG_WARN("findContentChildren tests use {} ms", TIMER_MSEC(findContentChildren));

}