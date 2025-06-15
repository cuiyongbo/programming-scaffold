#include "leetcode.h"

using namespace std;

/* leetcode: 167, 15, 16 */
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target);
    vector<vector<int>> threeSum(vector<int>& nums);
    int threeSumClosest(vector<int> &num, int target);
    int subarraysWithKDistinct(vector<int>& A, int K);
    int minSubArrayLen(int s, vector<int>& nums);
    int subarraySum(vector<int>& nums, int k);
};



/*
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
A subarray is a contiguous non-empty sequence of elements within an array.

Example 1:
    Input: nums = [1,1,1], k = 2
    Output: 2

Example 2:
    Input: nums = [1,2,3], k = 3
    Output: 2

Constraints:
    1 <= nums.length <= 2 * 104
    -1000 <= nums[i] <= 1000
    -107 <= k <= 107
*/
int Solution::subarraySum(vector<int>& nums, int k) {
    int ans = 0;
    int sum = 0;
    std::unordered_map<int, int> prefix_sum_count_map; // val, the number of subarrays whose sum is equal to val
    prefix_sum_count_map[0] = 1; // initialization
    for(auto n: nums) {
        sum += n;
        // given k=2, i=7, sum=5, prefix_sum_count_map[5-2]=4, so there are 4 subarrays whose sum are equal to k ending with nums[i]
        // since the sum is prefix sum, we always minus the sum of a subarray nums[0:k], the remaining array nums[k+1:i] is continuous
        ans += prefix_sum_count_map[sum-k];
        prefix_sum_count_map[sum]++;
    }
    return ans;
}


/*
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray whose sum ≥ s. If there isn’t one, return 0 instead.
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
        ans = min(ans, j-i); // why not j-i+1? because nums[j] is not inclusive when calculating sum
        sum -= nums[i]; // excluse nums[i] from next iteration, but we can reuse sum{nums[i+1:j]}
    }
    return ans==INT32_MAX ? 0 : ans;
}


/*
Given an array of integers that is already **sorted in ascending order**, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.
Please note that your returned answers (both index1 and index2) are one-based(1-indexed). You may assume that each input would have exactly one solution and you may not use the same element twice.

Given an input: numbers={2, 7, 11, 15}, target=9, Output: index1=1, index2=2
*/
vector<int> Solution::twoSum(vector<int>& nums, int target) {
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


/*
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero. Note: The solution set must not contain duplicate triplets.
For example, given array S = [-1, 0, 1, 2, -1, -4], A solution set is:
    [
        [-1, 0, 1],
        [-1, -1, 2]
    ]
*/
vector<vector<int>> Solution::threeSum(vector<int>& nums) {
    // sort `num` in ascending order
    std::sort(nums.begin(), nums.end(), std::less<int>());
    vector<vector<int>> ans;
    int sz = nums.size();
    for (int i=0; i<sz-2; ++i) {
        if (nums[i] > 0) { // prune invalid branches. because num[i+1:] must be greater than or equal to nums[i]
            break;
        }
        if (i>0 && nums[i] == nums[i-1]) { // skip duplicates
            continue;
        }
        // for each iteration, we fix i, and change l and r
        int l = i+1;
        int r = sz-1;
        while (l < r) { // NOTE that don't change this test to (l!=r)
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


/*
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. 
Return the sum of the three integers. You may assume that each input would have exactly one solution.
Example: Given array nums = [-1, 2, 1, -4], and target = 1. The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
*/
int Solution::threeSumClosest(vector<int>& nums, int target) {
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


/*
Given an integer array nums and an integer k, return the number of good subarrays of nums.
A good array is an array where the number of different integers in that array is exactly k.
For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3. A subarray is a contiguous part of an array.
Constraints:
    1 <= nums[i], k <= nums.length
Relative exercises:
    Longest Substring with At Most Two Distinct Characters
    Longest Substring with At Most K Distinct Characters
    Count Vowel Substrings of a String
    Number of Unique Flavors After Sharing K Candies
*/
int Solution::subarraysWithKDistinct(vector<int>& nums, int K) {
    // worker(k) means the number of subarrays with k or less than k distinct integer(s),
    // so the answer is worker(k)-worker(k-1)
    auto worker = [&] (int k) {
        int ans = 0;
        int sz = nums.size();
        // NOTE: we make sure that **1 <= nums[i], k <= nums.length**
        // count[i] means the count of i in `nums`
        vector<int> count(sz+1, 0);
        int distinct_nums = 0;
        for (int i=0,j=0; i<sz; ++i) {
            if (count[nums[i]] == 0) { // find a new distinct integer
                distinct_nums++;
            }
            count[nums[i]]++;
            assert(i>=j);
            // if there are more than `k` distinct numbers in nums[0:i]
            // shrink nums[j:i] so there are k or less than k distinct integer(s)
            for (; distinct_nums>k; j++) {
                --count[nums[j]];
                if (count[nums[j]] == 0) { // remove a distinct integer
                    distinct_nums--;
                }
            }
            assert(distinct_nums<=k);
            ans += (i-j+1); // it has to be this, not n*(n+1)/2, otherwise we will add the same case multiple times
            // nums[j:i] 1, 2, (i-j+1)
        }
        return ans;
    };
    return worker(K) - worker(K-1);
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


void subarraysWithKDistinct_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    int actual = ss.subarraysWithKDistinct(nums, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2,  expectedResult);
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


void subarraySum_scaffold(int input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input2);
    int actual = ss.subarraySum(nums, input1);
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

    SPDLOG_WARN("Running subarraysWithKDistinct tests:");
    TIMER_START(subarraysWithKDistinct);
    subarraysWithKDistinct_scaffold("[1,2,1,2,3]", 2, 7);
    subarraysWithKDistinct_scaffold("[1,2,1,3,4]", 3, 3);
    subarraysWithKDistinct_scaffold("[1,2,3,4]", 3, 2);
    TIMER_STOP(subarraysWithKDistinct);
    SPDLOG_WARN("subarraysWithKDistinct tests use {} ms", TIMER_MSEC(subarraysWithKDistinct));

    SPDLOG_WARN("Running minSubArrayLen tests:");
    TIMER_START(minSubArrayLen);
    minSubArrayLen_scaffold(7, "[2,3,1,2,4,3]", 2);
    minSubArrayLen_scaffold(6, "[2,3,1,2,4,3]", 2);
    minSubArrayLen_scaffold(4, "[2,3,1,2,4,3]", 1);
    minSubArrayLen_scaffold(9, "[2,3,1,2,4,3]", 3);
    TIMER_STOP(minSubArrayLen);
    SPDLOG_WARN("minSubArrayLen tests use {} ms", TIMER_MSEC(minSubArrayLen));

    SPDLOG_WARN("Running subarraySum tests:");
    TIMER_START(subarraySum);
    subarraySum_scaffold(2, "[1,1,1]", 2);
    subarraySum_scaffold(2, "[1,2,3]", 1);
    subarraySum_scaffold(3, "[1,2,3]", 2);
    TIMER_STOP(subarraySum);
    SPDLOG_WARN("subarraySum tests use {} ms", TIMER_MSEC(subarraySum));
}
