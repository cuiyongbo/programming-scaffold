#include "leetcode.h"

using namespace std;

/* leetcode: 295, 480, 239 */

namespace stray_dog {
class MedianFinder {
/*
    Find Median from Data Stream(the input stream may be unsorted) O(logn) + O(1)
    Median is the middle value in an ordered integer list. If the size of the list is even, 
    there is no middle value. So the median is the mean of the two middle value.

    Design a data structure that supports the following two operations:
        void addNum(int num) – Add a integer number from the data stream to the data structure.
        double findMedian() – Return the median of all elements so far.

    Related questions: 
        Sliding Window Median
        Finding MK Average
        Sequentially Ordinal Rank Tracker
*/
public:
    void addNum(int num) {
        m_left_subarray.push(num);
        m_right_subarray.push(m_left_subarray.top()); m_left_subarray.pop();
        if (m_right_subarray.size() > m_left_subarray.size()) {
            m_left_subarray.push(m_right_subarray.top());
            m_right_subarray.pop();
        }
    }
    double findMedian() {
        assert(m_left_subarray.size() >= m_right_subarray.size());
        if (m_right_subarray.size() == m_left_subarray.size()) {
            return (m_right_subarray.top() + m_left_subarray.top()) * 0.5;
        } else {
            return m_left_subarray.top();
        }
    }

private:
    /*
        assume there is a virtual array, consisting of m_left_subarray and m_right_subarray.
        according to the heap property:
            m_left_subarray.top() is the maximum element in the left subarray;
            m_right_subarray.top() is the minimum element in the right subarray;
        and we maintain the constraint that `m_left_subarray.size() >= m_right_subarray.size()`.
        so we can compute the median of the virtual array with the two heap tops in O(1)
    */
    priority_queue<int, vector<int>, less<int>> m_left_subarray; // max-heap
    priority_queue<int, vector<int>, greater<int>> m_right_subarray; // min-heap
};
}


void MedianFinder_scaffold(string operations, string args, string expectedOutputs) {
    SPDLOG_WARN("case({}) begin", operations);
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    stray_dog::MedianFinder tm;
    int n = operations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "addNum") {
            tm.addNum(std::stoi(funcArgs[i][0]));
            SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
        } else if (funcOperations[i] == "findMedian") {
            double actual = tm.findMedian();
            if (actual == std::stod(ans[i])) {
                SPDLOG_INFO("{}() passed", funcOperations[i]);
            } else {
                SPDLOG_ERROR("{}() failed, expectedResult={}, actual={}", funcOperations[i], ans[i], actual);
            }
        }
    }
}

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k);
    vector<int> maxSlidingWindow(vector<int>& nums, int k);
};


/*
Median is the middle value in an **ordered** integer list. 
If the size of the list is even, there is no middle value. 
So the median is the mean of the two middle value.

Given an **unsorted** array nums, there is a sliding window of size k 
which is moving from the very left of the array to the very right. 
You can only see the k numbers in the window. Each time the sliding window 
moves right by one position. Your job is to output the median array 
for each window in the original array.

related exercises:
    Minimize Malware Spread II
    Largest Unique Number
    Partition Array According to Given Pivot
*/
vector<double> Solution::medianSlidingWindow(vector<int>& nums, int k) {
    int sz = nums.size();
    int step = (k%2==1) ? 0 : 1;
    vector<double> ans; ans.reserve(sz-k+1); // the number of medians is equal to the number of subarrays
    multiset<int> windows(nums.begin(), nums.begin()+k);
    for (int i=k; i<=sz; ++i) { // NOTE that i is not inclusive when calculating the median of current subarray
        auto m1 = std::next(windows.begin(), (k-1)/2);
        auto m2 = std::next(m1, step); // if k is even, the median is the average of two middle values
        ans.push_back((*m1 + *m2)*0.5);
        if (i<sz) {
            windows.insert(nums[i]); // insert a new element
            // remove an element by its iterator.
            // DON'T call windows.erase by value, it will remove all elements with the same value
            windows.erase(windows.lower_bound(nums[i-k])); // remove the element going out of the window
        }
    }
    return ans;
}


/*
Given an array nums, there is a sliding window of size k which is moving 
from the very left of the array to the very right. You can only see the k 
numbers in the window. Each time the sliding window moves right by one position.

For example, Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       3
     1 [3  -1  -3] 5  3  6  7       3
     1  3 [-1  -3  5] 3  6  7       5
     1  3  -1 [-3  5  3] 6  7       5
     1  3  -1  -3 [5  3  6] 7       6
     1  3  -1  -3  5 [3  6  7]      7

Therefore, return the max sliding window as [3,3,5,5,6,7].
Note: You may assume k is always valid, ie: 1 ≤ k ≤ input array’s size for non-empty array.
*/
vector<int> Solution::maxSlidingWindow(vector<int>& nums, int k) {
if (1) { // binary tree solution
    int sz = nums.size();
    vector<int> ans; ans.reserve(sz-k+1);
    // for int type, the hash of int is itself
    // refer to [Why std::hash<int> seems to be identity function](https://stackoverflow.com/questions/38304877/why-stdhashint-seems-to-be-identity-function)
    multiset<int> windows(nums.begin(), nums.begin()+k);
    for (int i=k; i<=sz; ++i) {
        ans.push_back(*(windows.rbegin()));
        if (i<sz) {
            windows.insert(nums[i]); // add a new element
            windows.erase(windows.lower_bound(nums[i-k])); // remove the element going out of the window
        }
    }
    return ans;    
}

{ // priority_queue solution
    auto cmp = [&] (int l, int r) { return nums[l] < nums[r];};
    priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
    for (int i=0; i<k; ++i) {
        pq.push(i);
    }
    int sz = nums.size();
    vector<int> ans; ans.reserve(sz-k+1);
    for (int i=k; i<sz; ++i) {
        while (pq.top() < i-k) {
            pq.pop();
        }
        ans.push_back(nums[pq.top()]);
        pq.push(i);
    }
    ans.push_back(nums[pq.top()]);
    return ans;
}

}


void maxSlidingWindow_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.maxSlidingWindow(nums, input2);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, numberVectorToString(actual));
    }
}


void medianSlidingWindow_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    vector<double> expected = stringTo1DArray<double>(expectedResult);
    vector<double> actual = ss.medianSlidingWindow(nums, input2);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, numberVectorToString(actual));
    }
}


int main() {
    SPDLOG_WARN("Running MedianFinder tests:");
    TIMER_START(MedianFinder);
    MedianFinder_scaffold("[MedianFinder,addNum,findMedian,addNum,findMedian,addNum,findMedian]", 
                    "[[],[1],[],[3],[],[2],[]]",
                    "[null,null,1,null,2,null,2]");
    MedianFinder_scaffold("[MedianFinder,addNum,findMedian,addNum,findMedian,addNum,findMedian,addNum,findMedian,addNum,findMedian]", 
                    "[[],[-1],[],[-2],[],[-3],[],[-4],[],[-5],[]]",
                    "[null,null,-1.00000,null,-1.50000,null,-2.00000,null,-2.50000,null,-3.00000]");
    TIMER_STOP(MedianFinder);
    SPDLOG_WARN("MedianFinder tests use {} ms", TIMER_MSEC(MedianFinder));

    SPDLOG_WARN("Running medianSlidingWindow tests:");
    TIMER_START(medianSlidingWindow);
    medianSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 3, "[1,-1,-1,3,5,6]");
    medianSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 4, "[0,1,1,4,5.5]");
    medianSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 5, "[1,3,3,5]");
    TIMER_STOP(medianSlidingWindow);
    SPDLOG_WARN("medianSlidingWindow tests use {} ms", TIMER_MSEC(medianSlidingWindow));

    SPDLOG_WARN("Running maxSlidingWindow tests:");
    TIMER_START(maxSlidingWindow);
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 3, "[3,3,5,5,6,7]");
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 4, "[3,5,5,6,7]");
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 5, "[5,5,6,7]");
    TIMER_STOP(maxSlidingWindow);
    SPDLOG_WARN("maxSlidingWindow tests use {} ms", TIMER_MSEC(maxSlidingWindow));
}
