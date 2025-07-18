#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    vector<pair<int, int>> summaryRanges(vector<int>& nums);
    vector<vector<int>> merge(vector<vector<int>>& intervals);
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval);
    int findMinArrowShots(vector<vector<int>>& points);
};


/*
You are given a sorted unique integer array nums. A range [a,b] is the set of all integers from a to b (inclusive).
Return the smallest sorted list of ranges that cover all the numbers in the array exactly. That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:

"a->b" if a != b
"a" if a == b
 

Example 1:
Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"

Example 2:
Input: nums = [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: The ranges are:
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"
 
Constraints:
0 <= nums.length <= 20
-2^31 <= nums[i] <= 2^31 - 1
All the values of nums are unique.
nums is sorted in ascending order.
*/
vector<pair<int, int>> Solution::summaryRanges(vector<int>& nums) {
    vector<pair<int, int>> ans;
    ans.emplace_back(nums[0], nums[0]);
    for (int i=1; i<(int)nums.size(); i++) {
        if (ans.back().second+1 == nums[i]) {
            ans.back().second = nums[i];
        } else {
            ans.emplace_back(nums[i], nums[i]);
        }
    }
    return ans;
}


void summaryRanges_scaffold(string input1, string expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input1);
    auto actual = ss.summaryRanges(nums);
    auto expected = stringTo2DArray<int>(expectedResult);
    bool passed = false;
    if (actual.size() == expected.size()) {
        passed = true;
        for (int i=0; i<(int)actual.size(); i++) {
            if (actual[i].first != expected[i][0] || actual[i].second != expected[i][1]) {
                passed = false;
                break;
            }
        }
    }
    if (passed) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input1, expectedResult);
        for (auto p: actual) {
            printf("[%d,%d]", p.first, p.second);
        }
        cout << endl;
    }
}


/*
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
    Input: intervals = [[1,4],[4,5]]
    Output: [[1,5]]
    Explanation: Intervals [1,4] and [4,5] are considered overlapping.
*/
vector<vector<int>> Solution::merge(vector<vector<int>>& intervals) {
    // sort interval by left boundary then right boundary in ascending order
    std::sort(intervals.begin(), intervals.end(), [](const vector<int>& l, const vector<int>& r) {
        if (l[0] < r[0]) {
            return true;
        } else if (l[0] == r[0]) {
            return l[1] < r[1];
        } else {
            return false;
        }
    });
    vector<vector<int>> ans;
    ans.push_back(intervals[0]);
    for (int i=1; i<(int)intervals.size(); i++) {
        auto& b = ans.back(); // NOTE that it has to be a reference type
        if (b[1] < intervals[i][0]) { // not overlapped
            ans.push_back(intervals[i]);
        } else { // overlapped, merge two intervals. for case [1, 10], [4, 6]
            b[1] = std::max(b[1], intervals[i][1]);
        }
    }
    return ans;
}


void merge_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> intervals = stringTo2DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.merge(intervals);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for (const auto& row: actual) {
            cout << numberVectorToString<int>(row) << endl;
        }
    }
}


/*
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
Return intervals after the insertion.

Note that you don't need to modify intervals in-place. You can make a new array and return it.

Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
 

Constraints:
0 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 105
intervals is sorted by starti in ascending order.
newInterval.length == 2
0 <= start <= end <= 105
*/
vector<vector<int>> Solution::insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
    intervals.push_back(newInterval);
    return merge(intervals);
}


void insert_scaffold(string input, string input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> intervals = stringTo2DArray<int>(input);
    vector<int> new_interval = stringTo1DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.insert(intervals, new_interval);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual:", input, input2, expectedResult);
        for (const auto& row: actual) {
            cout << numberVectorToString<int>(row) << endl;
        }
    }
}


/*
There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array points where points[i] = [xstart, xend] denotes a balloon whose horizontal diameter stretches between xstart and xend. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up directly vertically (in the positive y-direction) from different points along the x-axis. A balloon with xstart and xend is burst by an arrow shot at x if xstart <= x <= xend. There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array points, return the minimum number of arrows that must be shot to burst all balloons.

Example 1:
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: The balloons can be burst by 2 arrows:
- Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
- Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].

Example 2:
Input: points = [[1,2],[3,4],[5,6],[7,8]]
Output: 4
Explanation: One arrow needs to be shot for each balloon for a total of 4 arrows.

Example 3:
Input: points = [[1,2],[2,3],[3,4],[4,5]]
Output: 2
Explanation: The balloons can be burst by 2 arrows:
- Shoot an arrow at x = 2, bursting the balloons [1,2] and [2,3].
- Shoot an arrow at x = 4, bursting the balloons [3,4] and [4,5].

Constraints:
1 <= points.length <= 105
points[i].length == 2
-2^31 <= xstart < xend <= 2^31 - 1
*/
int Solution::findMinArrowShots(vector<vector<int>>& points) {
    // sort interval by left boundary then right boundary in ascending order
    std::sort(points.begin(), points.end(), [](const vector<int>& l, const vector<int>& r) {
        if (l[0] < r[0]) {
            return true;
        } else if (l[0] == r[0]) {
            return l[1] < r[1];
        } else {
            return false;
        }
    });
    int ans = 1;
    int end = points[0][1];
    for (int i=1; i<(int)points.size(); i++) {
        if (points[i][0] <= end) {
            end = min(end, points[i][1]);
        } else {
            ans++;
            end = points[i][1];
        }
    }
    return ans;
}


void findMinArrowShots_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> intervals = stringTo2DArray<int>(input);
    int actual = ss.findMinArrowShots(intervals);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running summaryRanges tests: ");
    TIMER_START(summaryRanges);
    summaryRanges_scaffold("[0,1,2,4,5,7]", "[[0,2],[4,5],[7,7]]");
    summaryRanges_scaffold("[0,2,3,4,6,8,9]", "[[0,0],[2,4],[6,6],[8,9]]");
    TIMER_STOP(summaryRanges);
    SPDLOG_WARN("summaryRanges using {} ms", TIMER_MSEC(summaryRanges));

    SPDLOG_WARN("Running merge tests:");
    TIMER_START(merge);
    merge_scaffold("[[1,3],[2,6],[8,10],[15,18]]", "[[1,6],[8,10],[15,18]]");
    merge_scaffold("[[1,4],[4,5]]", "[[1,5]]");
    merge_scaffold("[[1,10],[4,5],[6, 8]]", "[[1,10]]");
    TIMER_STOP(merge);
    SPDLOG_WARN("merge tests use {} ms", TIMER_MSEC(merge));

    SPDLOG_WARN("Running insert tests:");
    TIMER_START(insert);
    insert_scaffold("[[1,3],[6,9]]", "[2,5]", "[[1,5],[6,9]]");
    insert_scaffold("[[1,2],[3,5],[6,7],[8,10],[12,16]]", "[4,8]", "[[1,2],[3,10],[12,16]]");
    TIMER_STOP(insert);
    SPDLOG_WARN("insert tests use {} ms", TIMER_MSEC(insert));

    SPDLOG_WARN("Running findMinArrowShots tests:");
    TIMER_START(findMinArrowShots);
    findMinArrowShots_scaffold("[[10,16],[2,8],[1,6],[7,12]]", 2);
    findMinArrowShots_scaffold("[[1,2],[3,4],[5,6],[7,8]]", 4);
    findMinArrowShots_scaffold("[[1,2],[2,3],[3,4],[4,5]]", 2);
    merge_scaffold("[[1,4],[4,5]]", "[[1,5]]");
    merge_scaffold("[[1,10],[4,5],[6, 8]]", "[[1,10]]");
    TIMER_STOP(findMinArrowShots);
    SPDLOG_WARN("findMinArrowShots tests use {} ms", TIMER_MSEC(findMinArrowShots));

}
