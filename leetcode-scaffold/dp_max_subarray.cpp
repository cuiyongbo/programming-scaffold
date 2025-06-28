#include "leetcode.h"

using namespace std;

/* leetcode: 53, 121, 309, 1013 */
class Solution {
public:
    int maxSubArray(vector<int>& nums);
    bool canThreePartsEqualSum(vector<int>& arr);
    int threePartsEqualSumCount(vector<int>& arr);
};


/*
Given an array of integers, return the number of solutions with which we can partition the array into three non-empty parts with equal sums.
Example:
    input: [0, 0, 0, 0]
    output: 3
*/
int Solution::threePartsEqualSumCount(vector<int>& arr) {
    int total = std::accumulate(arr.begin(), arr.end(), 0);
    int target = total/3;
    // test if sum(arr) can be divided by 3
    if (target*3 != total) {
        return 0;
    }
    int sz = arr.size();
    // dp[i] means the number of subarray[j:sz] whose sum is equal to target, i<=j<sz, in array[i:sz]
    vector<int> dp(sz, 0); 
    int count = 0;
    int cur_sum = 0; // suffix sum
    for (int i=sz-1; i>=0; --i) {
        cur_sum += arr[i];
        if (cur_sum == target) {
            count++;
        }
        dp[i] = count;
    }
    int ans = 0;
    cur_sum = 0; // prefix sum
    for (int i=0; i<sz-2; ++i) {
        cur_sum += arr[i];
        if (cur_sum == target) {
            ans += dp[i+2];
        }
    }
    return ans;
}


/*
Given an array of integers arr, return true if we can partition the array into three non-empty parts with equal sums.
Formally, we can partition the array if we can find indexes i + 1 < j with (arr[0] + arr[1] + ... + arr[i] == arr[i + 1] + arr[i + 2] + ... + arr[j - 1] == arr[j] + arr[j + 1] + ... + arr[arr.length - 1])
Hint: refer to canPartitionKSubsets exercise
*/
bool Solution::canThreePartsEqualSum(vector<int>& arr) {
    int total = std::accumulate(arr.begin(), arr.end(), 0);
    int sub = total/3;
    if (sub*3 != total) {
        return false;
    }
    int sz = arr.size();
    int i = 0;
    int left = 0;
    for (; i<sz; ++i) {
        left += arr[i];
        if (left == sub) { // stop iteration when left==sub
            break;
        }
    }
    int j = sz-1;
    int right = 0;
    for (; j>i; j--) {
        right += arr[j];
        if (right == sub) { // stop iteration when right==sub
            break;
        }
    }
    // there is at least one element between i and j,
    // and the sum of left and right parition is equal to sub
    return j-i>1 && left==sub && right==sub;
}



/*
Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.
*/
int Solution::maxSubArray(vector<int>& nums) {
// dp[i] means the largest sum of the contiguous subarray ending with nums[i]
// dp[i] = max(dp[i-1]+nums[i], nums[i])
{ // navie solution
    int ans = nums[0];
    int sz = nums.size();
    vector<int> dp = nums; // initialize trivial cases
    for (int i=1; i<sz; ++i) {
        dp[i] = max(dp[i-1]+nums[i], nums[i]);
        ans = max(ans, dp[i]);
    }
    return ans;
}

{ // solution with optimization of space usage
    int ans = nums[0];
    int a = nums[0];
    int n = nums.size();
    for (int i=1; i<n; ++i) {
        a = max(a+nums[i], nums[i]);
        ans = max(ans, a);
    }
    return ans;
}   

}


void maxSubArray_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> costs = stringTo1DArray<int>(input);
    int actual = ss.maxSubArray(costs);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void canThreePartsEqualSum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.canThreePartsEqualSum(prices);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void threePartsEqualSumCount_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.threePartsEqualSumCount(prices);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running maxSubArray tests:");
    TIMER_START(maxSubArray);
    maxSubArray_scaffold("[1]", 1);
    maxSubArray_scaffold("[1,-1,-1]", 1);
    maxSubArray_scaffold("[-2,1,-3,4,-1,2,1,-5,4]", 6);
    maxSubArray_scaffold("[5,4,-1,7,8]", 23);
    TIMER_STOP(maxSubArray);
    SPDLOG_WARN("maxSubArray tests use {} ms", TIMER_MSEC(maxSubArray));

    SPDLOG_WARN("Running canThreePartsEqualSum tests:");
    TIMER_START(canThreePartsEqualSum);
    canThreePartsEqualSum_scaffold("[0,2,1,-6,6,-7,9,1,2,0,1]", 1);
    canThreePartsEqualSum_scaffold("[0,2,1,-6,6,7,9,-1,2,0,1]", 0);
    canThreePartsEqualSum_scaffold("[3,3,6,5,-2,2,5,1,-9,4]", 1);
    canThreePartsEqualSum_scaffold("[18,12,-18,18,-19,-1,10,10]", 1);
    TIMER_STOP(canThreePartsEqualSum);
    SPDLOG_WARN("canThreePartsEqualSum tests use {} ms", TIMER_MSEC(canThreePartsEqualSum));

    SPDLOG_WARN("Running threePartsEqualSumCount tests:");
    TIMER_START(threePartsEqualSumCount);
    threePartsEqualSumCount_scaffold("[0,0,0,0]", 3);
    threePartsEqualSumCount_scaffold("[0,2,1,-6,6,-7,9,1,2,0,1]", 2);
    threePartsEqualSumCount_scaffold("[0,2,1,-6,6,7,9,-1,2,0,1]", 0);
    threePartsEqualSumCount_scaffold("[3,3,6,5,-2,2,5,1,-9,4]", 1);
    threePartsEqualSumCount_scaffold("[18,12,-18,18,-19,-1,10,10]", 1);
    threePartsEqualSumCount_scaffold("[1,1,1]", 1);
    threePartsEqualSumCount_scaffold("[1,1,1,1,1,1]", 1);
    threePartsEqualSumCount_scaffold("[1,1,1,1,1,1,1,1,1]", 1);
    threePartsEqualSumCount_scaffold("[1,1,1,1,1,1,1,1,1,1,1,1]", 1);
    TIMER_STOP(threePartsEqualSumCount);
    SPDLOG_WARN("threePartsEqualSumCount tests use {} ms", TIMER_MSEC(threePartsEqualSumCount));
}
