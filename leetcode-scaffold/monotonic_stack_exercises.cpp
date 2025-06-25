#include "leetcode.h"

using namespace std;

/* leetcode: 84, 496, 503, 901, 907, 1019 */
class Solution {
public:
    int largestRectangleArea(vector<int>& height);
    vector<int> nextGreaterElement_496(vector<int>& findNums, vector<int>& nums);
    vector<int> nextGreaterElement_503(vector<int>& nums);
    int nextGreaterElement_556(int n);
    vector<int> nextLargerNodes(ListNode* head);
    int sumSubarrayMins(vector<int>& A);
    int totalStrength(vector<int>& strength);
};


/*
Given a positive integer n, find the smallest integer which has exactly the same digits existing in the integer n and is greater in value than n. If no such positive integer exists, return -1. Note that the returned integer should fit in 32-bit integer, if there is a valid answer but it does not fit in 32-bit integer, return -1.

Example 1:
Input: n = 12
Output: 21

Example 2:
Input: n = 21
Output: -1
 
Constraints:
    1 <= n <= 2^31 - 1
*/
int Solution::nextGreaterElement_556(int n) {
    return -1;

}


/*
Given an array of integers A, find the sum of min(B), where B ranges over every (contiguous) subarray of A.
Since the answer may be large, return the answer modulo 10^9 + 7.

Example 1:
    Input: [3,1,2,4]
    Output: 17
    Explanation: Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4]. 
    Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.  Sum is 17.
*/
int Solution::sumSubarrayMins(vector<int>& A) {
    stack<pair<int, int>> st; // element value, frequency
    int sz = A.size();
    for (int i=0; i<sz; ++i) {
        st.emplace(A[i], 1);
        for (int j=i+1; j<sz; ++j) {
            // subarray: A[i, j]
            if(A[j] < st.top().first) { // find an element which is smaller than the last smallest one
                st.emplace(A[j], 1);
            } else /*if (A[j] >= st.top().first)*/ {
                st.top().second += 1;
            }
        }
    }
    int ans = 0;
    int modulo_num = 1e9 + 7;
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        ans = (ans + t.first * t.second) % modulo_num;
    }
    return ans;
}


/*
As the ruler of a kingdom, you have an army of wizards at your command.

You are given a 0-indexed integer array strength, where strength[i] denotes the strength of the ith wizard. For a contiguous group of wizards (i.e. the wizards' strengths form a subarray of strength), the total strength is defined as the product of the following two values:

The strength of the weakest wizard in the group.
The sum of all the individual strengths of the wizards in the group.

Return the sum of the total strengths of all contiguous groups of wizards. Since the answer may be very large, return it modulo 10^9 + 7.

A subarray is a contiguous non-empty sequence of elements within an array.
*/
int Solution::totalStrength(vector<int>& strength) {

    return 0;
}


/*
We are given a linked list with head as the first node.  Let’s number the nodes in the list: node_1, node_2, node_3, ... etc.

Each node may have a next larger value: for node_i, next_larger(node_i) is the node_j.val such that j > i, node_j.val > node_i.val, 
and j is the **smallest** possible choice. If such a j does not exist, the next larger value is 0.

Return an array of integers answer, where answer[i] = next_larger(node_{i+1}).

Note that in the example inputs below, arrays such as [2,1,5] represent the serialization of a 
linked list with a head node value of 2, second node value of 1, and third node value of 5.

Example 1:
Input: [2,1,5]
Output: [5,5,0]   
*/
vector<int> Solution::nextLargerNodes(ListNode* head) {
    // calculate the length of the link list
    int sz = 0;
    for (ListNode* p=head; p!=nullptr; p=p->next) {
        ++sz;
    }
    int idx=0;
    vector<int> ans(sz, 0);
    stack<pair<int, int>> st; // node val, node index
    for (ListNode* p=head; p!=nullptr; p=p->next) {
        while (!st.empty() && p->val>st.top().first) {
            ans[st.top().second] = p->val; st.pop();
        }
        st.emplace(p->val, idx); idx++;
    }
    return ans;
}


/*
Given a circular array (the next element of the last element is the first element of the array), 
print the Next Greater Number for every element. The Next Greater Number of a number x is the first greater
number to its traversing-order next in the array, which means you could search circularly to find its next greater number. 
If it doesn’t exist, output -1 for this number. for example, 

Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; The number 2 can't find next greater number; The second 1's next greater number needs to search circularly, which is also 2.
*/
vector<int> Solution::nextGreaterElement_503(vector<int>& nums) {
{ // refined solution
    /*
    nums: 1 2 3 4
    virtual array: 1 2 3 4 1 2 3 4
    left_index, right_index(not inclusive)
    0 - 4
    1 - 5
    2 - 6
    3 - 7
    */
    stack<int> st; // element index
    int sz = nums.size();
    vector<int> ans(sz, -1);
    for (int i=0; i<sz*2; i++) {
        int a_i = i%sz;
        while (!st.empty() && nums[a_i]>nums[st.top()]) {
            ans[st.top()] = nums[a_i]; st.pop();
        }
        st.push(a_i);
    }
    return ans; 
}

{ // naive solution
    stack<int> st; // element index
    int sz = nums.size();
    vector<int> ans(sz, -1);
    for (int i=0; i<sz; ++i) {
        while (!st.empty() && nums[i]>nums[st.top()]) {
            ans[st.top()] = nums[i]; st.pop();
        }
        st.push(i);
    }
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        for (int i=0; i<t; i++) { // find a possible larger element in naive way
            if (nums[i]>nums[t]) {
                ans[t] = nums[i];
                break;
            }
        }
    }
    return ans;
}

}


/*
You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements are subset of nums2. 
Find all the next greater numbers for nums1‘s elements in the corresponding places of nums2.

The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2. ([nextGreater of a in nums2 for a in nums1])
If it does not exist, output -1 for this number. for example,

Input: nums1 = [4,1,2], nums2 = [1,3,4,2]. 
Output: [-1,3,-1]
Explanation:
    For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
    For number 1 in the first array, the next greater number for it in the second array is 3.
    For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
*/
vector<int> Solution::nextGreaterElement_496(vector<int>& nums1, vector<int>& nums2) {
    stack<int> st;
    map<int, int> mp; // nums2[i], next greater element for nums2[i]
    int sz2 = nums2.size();
    for (int i=0; i<sz2; ++i) {
        while (!st.empty() && nums2[i] > st.top()) {
            mp[st.top()] = nums2[i]; st.pop();
        }
        st.push(nums2[i]);
    }
    int sz1 = nums1.size();
    vector<int> ans(sz1, -1);
    for (int i=0; i<sz1; ++i) {
        if (mp.find(nums1[i]) != mp.end()) {
            ans[i] = mp[nums1[i]];
        }
    }
    return ans;
}


/*
Given n non-negative integers representing the histogram’s bar height where the width of each bar is 1, 
find the area of largest rectangle in the histogram. for example, given height = [2,1,5,6,2,3], return 10.
*/
int Solution::largestRectangleArea(vector<int>& height) {
if (0) { // dp solution
    // dp[i] means largestRectangleArea ending with height[i]
    // dp[i] = max{(i-j+1)*min(height[j:i])} 0<=j<=i
    int ans = INT32_MIN;
    vector<int> dp = height; // initialization
    for (int i=0; i<(int)height.size(); i++) {
        int h = height[i];
        for (int j=i-1; j>=0; j--) {
            h = min(h, height[j]);
            dp[i] = max(dp[i], h*(i-j+1));
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}

if (0) { // refined solution
    stack<int> st;
    int sz = height.size();
    // aux[i] means if we use height[i] as the height, the rectangle wolud stretch over [left+1, right-1]
    vector<pair<int, int>> aux(sz, {-1, sz}); // (left, right), not inclusive
    for (int i=0; i<sz; ++i) {
        while (!st.empty() && height[st.top()] >= height[i]) {
            aux[st.top()].second = i;
            st.pop();
        }
        if (!st.empty()) { // set the left boundary for height[i]
            aux[i].first = st.top();
        }
        st.push(i);
    }
    int ans = INT32_MIN;
    for (int i=0; i<sz; ++i) {
        ans = max(ans, (aux[i].second-aux[i].first-1)*height[i]);
    }
    return ans;
}

{ // naive solution
    int sz = height.size();
    // aux[i] means if we use height[i] as the height, the rectangle wolud stretch over [left+1, right-1]
    vector<pair<int, int>> aux(sz, {-1, sz}); // (left, right), not inclusive
    stack<int> right;
    for (int i=0; i<sz; ++i) {
        // use `height[st.top()]` as the right boundary
        while (!right.empty() && height[right.top()] > height[i]) {
            aux[right.top()].second = i;
            right.pop();
        }
        right.push(i);
    }
    stack<int> left;
    for (int i=sz-1; i>=0; --i) {
        // use `height[st.top()]` as the left boundary
        while (!left.empty() && height[left.top()] > height[i]) {
            aux[left.top()].first = i;
            left.pop();
        }
        left.push(i);
    }
    int ans = INT32_MIN;
    for (int i=0; i<sz; ++i) {
        ans = max(ans, (aux[i].second-aux[i].first-1)*height[i]);
    }
    return ans;
}

}


void largestRectangleArea_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input);
    int actual = ss.largestRectangleArea(g);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void nextGreaterElement_496_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<int> nums1 = stringTo1DArray<int>(input1);
    vector<int> nums2 = stringTo1DArray<int>(input2);
    vector<int> ans = stringTo1DArray<int>(expectedResult);
    auto actual = ss.nextGreaterElement_496(nums1, nums2);
    if (actual == ans) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, numberVectorToString(actual));
    }
}


void nextGreaterElement_503_scaffold(string input1, string expectedResult) {
    Solution ss;
    vector<int> nums1 = stringTo1DArray<int>(input1);
    vector<int> ans = stringTo1DArray<int>(expectedResult);
    auto actual = ss.nextGreaterElement_503(nums1);
    if (actual == ans) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input1, expectedResult, numberVectorToString(actual));
    }
}


void nextLargerNodes_scaffold(string input1, string expectedResult) {
    Solution ss;
    ListNode* head = stringToListNode(input1);
    vector<int> ans = stringTo1DArray<int>(expectedResult);
    auto actual = ss.nextLargerNodes(head);
    if (actual == ans) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input1, expectedResult, numberVectorToString(actual));
    }
}


void sumSubarrayMins_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.sumSubarrayMins(nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

/*
    Write a class StockSpanner which collects daily price quotes for some stock, 
    and returns the span of that stock’s price for the current day.

    The span of the stock’s price today is defined as the maximum number 
    of **consecutive** days (starting from today and going backwards) for which 
    the price of the stock was less than or equal to today’s price.

    For example, if the price of a stock over the next 7 days were [100, 80, 60, 70, 60, 75, 85], 
    then the stock spans would be [1, 1, 1, 2, 1, 4, 6].

    For example, Input: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
    Output: [null,1,1,1,2,1,4,6]
    Explanation: 
        First, S = StockSpanner() is initialized.  Then:
            S.next(100) is called and returns 1,
            S.next(80) is called and returns 1,
            S.next(60) is called and returns 1,
            S.next(70) is called and returns 2,
            S.next(60) is called and returns 1,
            S.next(75) is called and returns 4,
            S.next(85) is called and returns 6.

    Note that (for example) S.next(75) returned 4, because the last 4 prices
    (including today's price of 75) were less than or equal to today's price.
    Note:
        Calls to StockSpanner.next(int price) will have 1 <= price <= 10^5.
        There will be at most 10000 calls to StockSpanner.next per test case.
        There will be at most 150000 calls to StockSpanner.next across all test cases.
        The total time limit for this problem has been reduced by 75% for C++, and 50% for all other languages.
*/

class StockSpanner {
public:
    StockSpanner() {}
    int next(int price) {
        auto p = std::make_pair(price, 1);
        while (!m_st.empty() && m_st.top().first<=price) {
            p.second += m_st.top().second; m_st.pop();
        }
        m_st.push(p);
        return p.second;
    }

private:
    stack<pair<int, int>> m_st; // price, span
};


void StockSpanner_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    StockSpanner tm;
    int n = (int)ans.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "next") {
            int actual = tm.next(std::stoi(funcArgs[i][0]));
            if (actual == std::stoi(ans[i])) {
                SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
            } else {
                SPDLOG_ERROR("{}({}) failed, expected={}, actual={}", funcOperations[i], funcArgs[i][0], ans[i], actual);
            }
        }
    }
}


int main() {
    SPDLOG_WARN("Running largestRectangleArea tests:");
    TIMER_START(largestRectangleArea);
    largestRectangleArea_scaffold("[1,3,2,4]", 6);
    largestRectangleArea_scaffold("[3,2,1,2,1,2,1,2,1]", 9);
    largestRectangleArea_scaffold("[3,2,0,2,0,2,0,2,1]", 4);
    largestRectangleArea_scaffold("[2,1,5,6,2,3]", 10);
    largestRectangleArea_scaffold("[2,4]", 4);
    largestRectangleArea_scaffold("[2,2,2,2,2,2,2]", 14);
    TIMER_STOP(largestRectangleArea);
    SPDLOG_WARN("largestRectangleArea tests use {} ms", TIMER_MSEC(largestRectangleArea));

    SPDLOG_WARN("Running nextGreaterElement_496 tests:");
    TIMER_START(nextGreaterElement_496);
    nextGreaterElement_496_scaffold("[4,1,2]", "[1,3,2,4]", "[-1,3,4]");
    nextGreaterElement_496_scaffold("[4,1,2]", "[1,3,4,2]", "[-1,3,-1]");
    nextGreaterElement_496_scaffold("[2,4]", "[1,2,3,4]", "[3,-1]");
    TIMER_STOP(nextGreaterElement_496);
    SPDLOG_WARN("nextGreaterElement_496 tests use {} ms", TIMER_MSEC(nextGreaterElement_496));

    SPDLOG_WARN("Running nextGreaterElement_503 tests:");
    TIMER_START(nextGreaterElement_503);
    nextGreaterElement_503_scaffold("[4,3,2,1]", "[-1,4,4,4]");
    nextGreaterElement_503_scaffold("[4,3,2]", "[-1,4,4]");
    nextGreaterElement_503_scaffold("[4,1,2]", "[-1,2,4]");
    nextGreaterElement_503_scaffold("[2,4]", "[4, -1]");
    nextGreaterElement_503_scaffold("[1,2,1]", "[2, -1, 2]");
    nextGreaterElement_503_scaffold("[1,2,3,4,3]", "[2,3,4,-1,4]");
    TIMER_STOP(nextGreaterElement_503);
    SPDLOG_WARN("nextGreaterElement_503 tests use {} ms", TIMER_MSEC(nextGreaterElement_503));

    SPDLOG_WARN("Running nextLargerNodes tests:");
    TIMER_START(nextLargerNodes);
    nextLargerNodes_scaffold("[2,1,5]", "[5,5,0]");
    nextLargerNodes_scaffold("[2,7,4,3,5]", "[7,0,5,5,0]");
    nextLargerNodes_scaffold("[1,7,5,1,9,2,5,1]", "[7,9,9,9,0,5,0,0]");
    nextLargerNodes_scaffold("[5,4,3,2,1]", "[0,0,0,0,0]");
    TIMER_STOP(nextLargerNodes);
    SPDLOG_WARN("nextLargerNodes tests use {} ms", TIMER_MSEC(nextLargerNodes));

    SPDLOG_WARN("Running sumSubarrayMins tests:");
    TIMER_START(sumSubarrayMins);
    sumSubarrayMins_scaffold("[3,1,2,4]", 17);
    TIMER_STOP(sumSubarrayMins);
    SPDLOG_WARN("sumSubarrayMins tests use {} ms", TIMER_MSEC(sumSubarrayMins));

    SPDLOG_WARN("Running StockSpanner tests:");
    TIMER_START(StockSpanner);
    StockSpanner_scaffold("[StockSpanner,next,next,next,next,next,next,next]", 
                    "[[],[100],[80],[60],[70],[60],[75],[85]]",
                    "[null,1,1,1,2,1,4,6]");
    TIMER_STOP(StockSpanner);
    SPDLOG_WARN("StockSpanner tests use {} ms", TIMER_MSEC(StockSpanner));

    return 0;
}
