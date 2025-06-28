#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);
    int removeElement(vector<int>& nums, int val);
    int removeDuplicates(vector<int>& nums);
    int removeDuplicates_80(vector<int>& nums);
    int majorityElement(vector<int>& nums);
    void rotate(vector<int>& nums, int k);
    // for <Best time to buy and sell stock> problems
    // stock_exercises.cpp
    bool canJump(vector<int>& nums);
    int jump(vector<int>& nums);
    int hIndex(vector<int>& citations);
    vector<int> productExceptSelf(vector<int>& nums);
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost);
};


/*
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.


Example 1:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.

Example 2:
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].

Example 3:
Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.

Constraints:

nums1.length == m + n
nums2.length == n
0 <= m, n <= 200
1 <= m + n <= 200
-109 <= nums1[i], nums2[j] <= 109

Hint: since we need use nums1 as merged result and it has been pre-allocated, we can merge two array from right to left, and take the cacant places first. it is similar to what memset does when moving memories which are overlapped
*/
void Solution::merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int total = m+n;
    int i = m-1;
    int j = n-1;
    // iterate nums1, nums2 from right to left
    for (int k=total-1; k>=0; k--) {
        if (j<0 || ((i>=0)&&(nums1[i]>nums2[j]))) {
            nums1[k] = nums1[i]; i--;
        } else {
            nums1[k] = nums2[j]; j--;
        }
    }
}


void merge_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    auto nums1 = stringTo1DArray<int>(input1);
    auto nums2 = stringTo1DArray<int>(input2);
    auto expected = stringTo1DArray<int>(expectedResult);
    ss.merge(nums1, nums1.size()-nums2.size(), nums2, nums2.size());
    if (nums1 == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult: {}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult: {}) failed, actual: {}", input1, input2, expectedResult, numberVectorToString(nums1));
    }
}


/*
Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
Return k.

Example 1:
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).

Constraints:

1 <= nums.length <= 3 * 104
-104 <= nums[i] <= 104
nums is sorted in non-decreasing order.
*/
int Solution::removeElement(vector<int>& nums, int val) {
    // pre-condition: nums[0:k] contains elements which are not equal to val
    // invariant: nums[0:k] contains elements which are not equal to val
    // post-condition: nums[0:k+1] contains elements which are not equal to val
    int k = -1;
    for (int i=0; i<(int)nums.size(); i++) {
        if (nums[i] != val) {
            k++;
            nums[k] = nums[i];
        }
    }
    k++;
    return k;
}


void removeElement_scaffold(string input1, int input2) {
    Solution ss;
    auto nums1 = stringTo1DArray<int>(input1);
    int k = ss.removeElement(nums1, input2);
    bool passed = true;
    for (int i=0; i<k; i++) {
        if (nums1[i] == input2) {
            passed = false;
            break;
        }
    }
    if (passed) {
        SPDLOG_INFO("Case({}, {}) passed", input1, input2);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: {}", input1, input2, numberVectorToString(nums1));
    }
}


/*
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same. Then return the number of unique elements in nums.

Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially. The remaining elements of nums are not important as well as the size of nums.
Return k.


Example 1:
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
*/
int Solution::removeDuplicates(vector<int>& nums) {
    if (nums.empty()) {
        return 0;
    }
    int k = 0;
    for (int i=1; i<(int)nums.size(); i++) {
        if (nums[i] != nums[k]) {
            k++;
            nums[k] = nums[i];
        }
    }
    k++;
    return k;
}


/*
Follow up: what if we allow that each unique element appears at most twice?

Example 1:
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]

Example 2:
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
*/
int Solution::removeDuplicates_80(vector<int>& nums) {
    if (nums.empty()) {
        return 0;
    }
    int k = 0;
    for (int i=1; i<(int)nums.size(); i++) {
        // at most one duplicate for each element
        if (k>0 && (nums[k-1]==nums[i])) {
            continue;
        }
        k++;
        nums[k] = nums[i];
    }
    k++;
    return k;
}


void removeDuplicates_scaffold(string input1, string input2) {
    Solution ss;
    auto nums1 = stringTo1DArray<int>(input1);
    auto nums2 = stringTo1DArray<int>(input2);
    int k = ss.removeDuplicates(nums1);
    bool passed = true;
    for (int i=0; i<k; i++) {
        if (nums1[i] != nums2[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        SPDLOG_INFO("Case({}, {}) passed", input1, input2);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: {}", input1, input2, numberVectorToString(nums1));
    }
}


/*
Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
You may assume that the array is non-empty and the majority element always exist in the array.
Hint: Moore Voting Algorithm

Follow up: leetcode 229

Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

Hint: Think about the possible number of elements that can appear more than ⌊ n/3 ⌋ times in the array.  It can be at most two. Why? Consider using Boyer-Moore Voting Algorithm, which is efficient for finding elements that appear more than a certain threshold.
*/
int Solution::majorityElement(vector<int>& nums) {
    int m = 0;
    int cnt = 0;
    for (auto n: nums) {
        if (cnt == 0) {
            m = n;
            cnt = 1;
        } else {
            if (m==n) {
                cnt++;
            } else {
                cnt--; // since the majority element occurs more than [n/2] times, its occurrences would be greater than 0 at last
            }
        }
    }
    SPDLOG_INFO("ans={}, count={}", m, cnt);
    return m;
}


void majorityElement_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.majorityElement(nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, actual);
    }
}


/*
Given an array, rotate the array to the right by k steps, where k is non-negative.
Example 1:
    Input: nums=[1,2,3,4,5,6,7], k=3
    Output: [5,6,7,1,2,3,4]
    Explanation:
    rotate 1 steps to the right: [7,1,2,3,4,5,6]
    rotate 2 steps to the right: [6,7,1,2,3,4,5]
    rotate 3 steps to the right: [5,6,7,1,2,3,4]
Example 2:
    Input: nums = [-1,-100,3,99], k = 2
    Output: [3,99,-1,-100]
*/
void Solution::rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k %= n; // k may be larger than array_size, and we don't need to swap more than n steps
    if (k==0) { // no need to perform further operation
        return;
    }

if (1) { // std solution
    // right is not inclusive
    std::reverse(nums.begin(), nums.end());
    std::reverse(nums.begin(), nums.begin()+k);
    std::reverse(nums.begin()+k, nums.end());

    /*
    you may also to reverse nums in following orders:
    std::reverse(nums.begin(), nums.begin()+n-k);
    std::reverse(nums.begin()+n-k, nums.end());
    std::reverse(nums.begin(), nums.end());
    */
    return;
}

{ // naive solution
    // reverse subarray nums[s:e], e is not inclusive
    auto reverse_worker = [&] (int s, int e) {
        for (int i=0; i<(e-s); ++i) {
            if (s+i >= e-i-1) {
                break;
            }
            swap(nums[s+i], nums[e-i-1]);
        }
    };
    // 1. reverse left part
    reverse_worker(0, n-k);
    // 2. reverse right part
    reverse_worker(n-k, n);
    // 3. then reverse the whole array
    reverse_worker(0, n);
}
}


void rotate_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    vector<int> v2 = stringTo1DArray<int>(expectedResult);
    ss.rotate(v1, input2);
    if(v1 == v2) {
        SPDLOG_INFO( "Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR( "Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, numberVectorToString(v1));
    }
}


/*
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position. Return true if you can reach the last index, or false otherwise.

Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
 
Constraints:
1 <= nums.length <= 104
0 <= nums[i] <= 105
*/
bool Solution::canJump(vector<int>& nums) {
    int mx = 0; // the farthest index we can reach
    for (int i=0; i<(int)nums.size(); i++) {
        if (mx < i) { // oh, we cann't reach index i
            return false;
        }
        mx = max(mx, i+nums[i]);
    }
    return true;
}


void canJump_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    bool actual = ss.canJump(nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, actual);
    }
}


/*
You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i+j] where: 0<=j<=nums[i] and i+j<n.
Return the minimum number of jumps to reach nums[n-1]. The test cases are generated such that you can reach nums[n-1].


Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2

Constraints:

1 <= nums.length <= 104
0 <= nums[i] <= 1000
It's guaranteed that you can reach nums[n - 1].
*/
int Solution::jump(vector<int>& nums) {
    int ans = 0;
    int mx = 0; // the farthest index we can reach
    int last = 0; // last position, if we reach mx, then we need a jump
    for (int i = 0; i < (int)nums.size()-1; ++i) {
        mx = max(mx, i + nums[i]);
        if (last == i) {
            ++ans;
            last = mx;
        }
    }
    return ans;
}


void jump_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.jump(nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, actual);
    }
}


/*
Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper, return the researcher's h-index.

According to the definition of h-index on Wikipedia: The h-index is defined as the maximum value of h such that the given researcher has published at least h papers that have each been cited at least h times.

Input: citations = [3,0,6,1,5]
Output: 3
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively.
Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, their h-index is 3.
Example 2:

Input: citations = [1,3,1]
Output: 1
 

Constraints:

n == citations.length
1 <= n <= 5000
0 <= citations[i] <= 1000
*/
int Solution::hIndex(vector<int>& citations) {
    // sort citations in ascending order
    std::sort(citations.begin(), citations.end(), std::less<int>());
    int ans = 0;
    int sz = citations.size();
    for(int i=0; i<sz; ++i) {
        // The h-index is defined as the maximum value of h such that the given researcher has published at least h papers that have each been cited at least h times.
        if(citations[i] >= sz-i) {
            ans = sz-i;
            break;
        }
    }
    return ans;
}


void hIndex_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.hIndex(nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, actual);
    }
}


/*
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

Constraints:
2 <= nums.length <= 105
-30 <= nums[i] <= 30
The input is generated such that answer[i] is guaranteed to fit in a 32-bit integer.
*/
vector<int> Solution::productExceptSelf(vector<int>& nums) {
    int sz = nums.size();
    vector<int> ans(sz, 1);
    int left = 1;
    for (int i=0; i<sz; i++) {
        ans[i] *= left;
        left *= nums[i];
    }
    int right = 1;
    for (int i=sz-1; i>=0; i--) {
        ans[i] *= right;
        right *= nums[i];
    }
    return ans;
}


void productExceptSelf_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    auto actual = ss.productExceptSelf(nums);
    if (actual == expected) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, numberVectorToString(actual));
    }
}


/*
There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique.

Example 1:
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.

Example 2:
Input: gas = [2,3,4], cost = [3,4,3]
Output: -1
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.
 
Constraints:
n == gas.length == cost.length
1 <= n <= 105
0 <= gas[i], cost[i] <= 104
*/
int Solution::canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
if (0) { // naive solution, Time Limit Exceeded
    int sz = gas.size();
    for (int i=0; i<sz; i++) {
        bool ok = true;
        int remaining_gas = 0;
        for (int j=i; j<i+sz; j++) {
            int aj = j%sz;
            remaining_gas = remaining_gas + gas[aj] - cost[aj];
            if (remaining_gas < 0) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return i;
        }
    }
    return -1;
}

{
    int total=0, sum=0, start=0;
    int sz = gas.size();
    for (int i=0; i<sz; i++) {
        total += gas[i] - cost[i];
        sum += gas[i] - cost[i];
        if (sum < 0) { // we can not take stations in [start, i] as start
            start = i+1;
            sum = 0;
        }
    }
    return (total<0) ? -1 : start;
}

}


void canCompleteCircuit_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    auto nums1 = stringTo1DArray<int>(input1);
    auto nums2 = stringTo1DArray<int>(input2);
    int actual = ss.canCompleteCircuit(nums1, nums2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult: {}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult: {}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}



int main() {
    SPDLOG_WARN("Running merge tests: ");
    TIMER_START(merge);
    merge_scaffold("[1,2,3,0,0,0]", "[2,5,6]", "[1,2,2,3,5,6]");
    merge_scaffold("[1]", "[]", "[1]");
    merge_scaffold("[0]", "[1]", "[1]");
    TIMER_STOP(merge);
    SPDLOG_WARN("merge using {} ms", TIMER_MSEC(merge));

    SPDLOG_WARN("Running removeElement tests: ");
    TIMER_START(removeElement);
    removeElement_scaffold("[1,2,3,0,0,0]", 0);
    removeElement_scaffold("[1]", 1);
    removeElement_scaffold("[1,1]", 1);
    removeElement_scaffold("[0]", 1);
    removeElement_scaffold("[]", 1);
    removeElement_scaffold("[3,2,2,3]", 3);
    removeElement_scaffold("[3,2,2,3]", 2);
    removeElement_scaffold("[0,1,2,2,3,0,4,2]", 2);
    TIMER_STOP(removeElement);
    SPDLOG_WARN("removeElement using {} ms", TIMER_MSEC(removeElement));

    SPDLOG_WARN("Running removeDuplicates tests: ");
    TIMER_START(removeDuplicates);
    removeDuplicates_scaffold("[1,2,3,0,0,0]", "[1,2,3,0]");
    removeDuplicates_scaffold("[1]", "[1]");
    removeDuplicates_scaffold("[1,1]", "[1]");
    removeDuplicates_scaffold("[]", "[]");
    removeDuplicates_scaffold("[0,0,1,1,1,2,2,3,3,4]", "[0,1,2,3,4]");
    removeDuplicates_scaffold("[1,1,2]", "[1,2]");
    TIMER_STOP(removeDuplicates);
    SPDLOG_WARN("removeDuplicates using {} ms", TIMER_MSEC(removeDuplicates));

    SPDLOG_WARN("Running majorityElement tests:");
    TIMER_START(majorityElement);
    majorityElement_scaffold("[6,1,2,8,6,4,5,3,6,6,6,5,6,6,6,6]", 6);
    majorityElement_scaffold("[6]", 6);
    majorityElement_scaffold("[6,1,6]", 6);
    // majorityElement_scaffold("[6,1]", -1);
    TIMER_STOP(majorityElement);
    SPDLOG_WARN("majorityElement tests use {} ms", TIMER_MSEC(majorityElement));

    SPDLOG_WARN("Running rotate tests:");
    TIMER_START(rotate);
    rotate_scaffold("[1,2,3,4,5,6,7]", 3, "[5,6,7,1,2,3,4]");
    rotate_scaffold("[1,2,3,4,5,6,7]", 4, "[4,5,6,7,1,2,3]");
    rotate_scaffold("[-1,-100,3,99]", 2, "[3,99,-1,-100]");
    TIMER_STOP(rotate);
    SPDLOG_WARN("rotate tests use {} ms", TIMER_MSEC(rotate));

    SPDLOG_WARN("Running canJump tests:");
    TIMER_START(canJump);
    canJump_scaffold("[2,3,1,1,4]", true);
    canJump_scaffold("[3,2,1,0,4]", false);
    TIMER_STOP(canJump);
    SPDLOG_WARN("canJump tests use {} ms", TIMER_MSEC(canJump));

    SPDLOG_WARN("Running jump tests:");
    TIMER_START(jump);
    jump_scaffold("[2,3,1,1,4]", 2);
    jump_scaffold("[2,3,0,1,4]", 2);
    TIMER_STOP(jump);
    SPDLOG_WARN("jump tests use {} ms", TIMER_MSEC(jump));

    SPDLOG_WARN("Running hIndex tests:");
    TIMER_START(hIndex);
    hIndex_scaffold("[3,0,6,1,5]", 3);
    hIndex_scaffold("[1,3,1]", 1);
    hIndex_scaffold("[100]", 1);
    hIndex_scaffold("[0]", 0);
    hIndex_scaffold("[0,0]", 0);
    hIndex_scaffold("[0,0,0]", 0);
    TIMER_STOP(hIndex);
    SPDLOG_WARN("hIndex tests use {} ms", TIMER_MSEC(hIndex));

    SPDLOG_WARN("Running productExceptSelf tests: ");
    TIMER_START(productExceptSelf);
    productExceptSelf_scaffold("[1,2,3,4]", "[24,12,8,6]");
    productExceptSelf_scaffold("[-1,1,0,-3,3]", "[0,0,9,0,0]");
    TIMER_STOP(productExceptSelf);
    SPDLOG_WARN("removeDuplicates using {} ms", TIMER_MSEC(productExceptSelf));

    SPDLOG_WARN("Running canCompleteCircuit tests: ");
    TIMER_START(canCompleteCircuit);
    canCompleteCircuit_scaffold("[1,2,3,4,5]", "[3,4,5,1,2]", 3);
    canCompleteCircuit_scaffold("[2,3,4]", "[3,4,3]", -1);
    TIMER_STOP(canCompleteCircuit);
    SPDLOG_WARN("canCompleteCircuit using {} ms", TIMER_MSEC(canCompleteCircuit));

}
