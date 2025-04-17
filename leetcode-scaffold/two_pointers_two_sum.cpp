#include "leetcode.h"

using namespace std;

/* leetcode: 167, 15, 16 */
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target);
    vector<vector<int>> threeSum(vector<int>& nums);
    int threeSumClosest(vector<int> &num, int target);
};


vector<int> Solution::twoSum(vector<int>& nums, int target) {
/*
    Given an array of integers that is already **sorted in ascending order**, find two numbers such that they add up to a specific target number.

    The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.
    Please note that your returned answers (both index1 and index2) are one-based. You may assume that each input would have exactly one solution and you may not use the same element twice.

    Given an input: numbers={2, 7, 11, 15}, target=9, Output: index1=1, index2=2
*/
    int l=0;
    int r=nums.size()-1;
    while (l != r) { // brilliant
        int m = nums[l]+nums[r];
        if (m == target) {
            break;
        } else if (m < target) {
            ++l;
        } else {
            --r;
        }
    }
    return {l+1, r+1};
}

vector<vector<int>> Solution::threeSum(vector<int>& nums) {
/*
    Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
    Find all unique triplets in the array which gives the sum of zero. Note: The solution set must not contain duplicate triplets.
    For example, given array S = [-1, 0, 1, 2, -1, -4], A solution set is:
        [
            [-1, 0, 1],
            [-1, -1, 2]
        ]
*/
    // sort `num` in ascending order
    std::sort(nums.begin(), nums.end(), std::less<int>());
    //print_vector(nums);
    vector<vector<int>> ans;
    int sz = nums.size();
    for (int i=0; i<sz-2; ++i) {
        if (nums[i] > 0) { // prune useless branches. because num[i+1:] must be greater than or equal to nums[i]
            break;
        }
        if (i>0 && nums[i] == nums[i-1]) { // skip duplicates
            continue;
        }
        // for each iteration, we fix i, and change l and r
        int l = i+1;
        int r = sz-1;
        while (l < r) {
            int m = nums[i] + nums[l] + nums[r];
            //printf("i=%d, l=%d, r=%d, m=%d\n", i, l, r, m);
            if (m == 0) {
                ans.push_back({nums[i], nums[l], nums[r]});
                l++; r--;
                while (l<r && nums[l] == nums[l-1]) { // skip duplicates
                    ++l;
                }
                while (l<r && nums[r] == nums[r+1]) { // skip duplicates
                    --r;
                }
            } else if (m < 0) {
                ++l;
            } else {
                --r;
            }
        }
    }
    return ans;
}

int Solution::threeSumClosest(vector<int>& nums, int target) {
/*
    Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. 
    Return the sum of the three integers. You may assume that each input would have exactly one solution.
    Example: Given array nums = [-1, 2, 1, -4], and target = 1. The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
*/
    // sort nums in ascending order
    std::sort(nums.begin(), nums.end(), std::less<int>());
    int ans = 0;
    int diff = INT32_MAX;
    int sz = nums.size();
    for (int i=0; i<sz-2; ++i) {
        // fix i at iteration
        int l = i+1;
        int r = sz-1;
        while (l < r) {
            int m = nums[i] + nums[l] + nums[r];
            if (diff > abs(m-target)) {
                diff = abs(m-target);
                ans = m;
            }
            if (m == target) {
                return target;
            } else if (m < target) {
                ++l;
            } else {
                --r;
            }
        }
    }
    return ans;
}


void twoSum_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.twoSum(A, input2);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, numberVectorToString<int>(actual));
    }
}


void threeSum_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.threeSum(A);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for (const auto& s: actual) {
            cout << numberVectorToString<int>(s) << endl;
        }
    }
}


void threeSumClosest_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    int actual = ss.threeSumClosest(A, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running twoSum tests:");
    TIMER_START(twoSum);
    twoSum_scaffold("[1,2]", 3, "[1,2]");
    twoSum_scaffold("[2, 7, 11, 15]", 9, "[1,2]");
    TIMER_STOP(twoSum);
    SPDLOG_WARN("twoSum tests use {} ms", TIMER_MSEC(twoSum));

    SPDLOG_WARN("Running threeSum tests:");
    TIMER_START(threeSum);
    threeSum_scaffold("[-1,0,1,2,-1,-4]", "[[-1,-1,2],[-1,0,1]]");
    threeSum_scaffold("[0,0,0]", "[[0,0,0]]");
    threeSum_scaffold("[0,0,0,0]", "[[0,0,0]]");
    threeSum_scaffold("[0,0,0,0,0]", "[[0,0,0]]");
    threeSum_scaffold("[0,0,0,0,0,0]", "[[0,0,0]]");
    TIMER_STOP(threeSum);
    SPDLOG_WARN("threeSum tests use {} ms", TIMER_MSEC(threeSum));

    SPDLOG_WARN("Running threeSumClosest tests:");
    TIMER_START(threeSumClosest);
    threeSumClosest_scaffold("[-1, 2, 1, -4]", 1, 2);
    threeSumClosest_scaffold("[1,1,1,1]", 1, 3);
    threeSumClosest_scaffold("[1,1,1,1]", 2, 3);
    threeSumClosest_scaffold("[1,1,1,1]", 3, 3);
    threeSumClosest_scaffold("[1,1,1,1]", 4, 3);
    TIMER_STOP(threeSumClosest);
    SPDLOG_WARN("threeSumClosest tests use {} ms", TIMER_MSEC(threeSumClosest));
}
