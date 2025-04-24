#include "leetcode.h"

using namespace std;

/* leetcode: 150, 3, 223, 836, 189, 56 */
class Solution {
public:
    int lengthOfLongestSubstring(string s);
    int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2);
    bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2);
    void rotate(vector<int>& nums, int k);
    vector<vector<int>> merge(vector<vector<int>>& intervals);
};


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
    stack<vector<int>> st;
    for (const auto& p: intervals) {
        if (st.empty()) {
            st.push(p);
        } else {
            // if overlapped, merge two intervals
            // otherwise save a new interval
            auto& t = st.top();
            if (t[1] < p[0]) {
                st.push(p);
            } else {
                t[1] = p[1];
            }
        }
    }
    vector<vector<int>> ans;
    while (!st.empty()) {
        ans.push_back(st.top()); st.pop();
    }
    std::reverse(ans.begin(), ans.end());
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
Given an array, rotate the array to the right by k steps, where k is non-negative.
Example 1:
    Input: nums = [1,2,3,4,5,6,7], k = 3
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
    // reverse subarray nums[s:e], e is not inclusive
    auto reverse_worker = [&] (int s, int e) {
        for (int i=0; i<(e-s); ++i) {
            if (s+i >= e-i-1) {
                break;
            }
            swap(nums[s+i], nums[e-i-1]);
        }
    };
    int n = nums.size();
    k %= n; // k may be larger than array_size
    // 1. reverse left part
    reverse_worker(0, n-k);
    // 2. reverse right part
    reverse_worker(n-k, n);
    // 3. then reverse the whole array
    reverse_worker(0, n);
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
An axis-aligned rectangle is represented as a list [x1, y1, x2, y2], where (x1, y1) is the coordinate of its bottom-left corner, 
and (x2, y2) is the coordinate of its top-right corner. Its top and bottom edges are parallel to the X-axis, and its left and right edges are parallel to the Y-axis.
Two rectangles overlap if the area of their intersection is positive. To be clear, two rectangles that only touch at the corner or edges do not overlap.
Given two axis-aligned rectangles rec1 and rec2, return true if they overlap, otherwise return false.
*/
bool Solution::isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
    if (rec1[0] >= rec2[2] || rec2[0] >= rec1[2] || // separated from x-axis
        rec1[1] >= rec2[3] || rec2[1] >= rec1[3] ) { // separated from y-axis
        return false;
    } else {
        return true;
    }
}


/*
Given the coordinates of two rectilinear rectangles in a 2D plane, return the total area covered by the two rectangles.
The first rectangle is defined by its bottom-left corner (ax1, ay1) and its top-right corner (ax2, ay2).
The second rectangle is defined by its bottom-left corner (bx1, by1) and its top-right corner (bx2, by2).
Hint: ans = area1 + area2 - joint_area
*/
int Solution::computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
    auto area = [] (int ax1, int ay1, int ax2, int ay2) {
        return (ax2-ax1)*(ay2-ay1);
    };
    int ans = area(ax1, ay1, ax2, ay2) + area(bx1, by1, bx2, by2);
    int intersection = 0;
    if (ax1>=bx2 || bx1>=ax2 || ay1>=by2 || by1>=ay2) { // the two rectangles are not overlapped
        intersection = 0;
    } else { // brilliant
        intersection = area(max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2));
    }
    return ans - intersection;
}


/*
Given a string s, find the length of the longest substring(not subsequence) without repeating characters.

Example 1:
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.
*/
int Solution::lengthOfLongestSubstring(string str) {
    int ans = 0;
    map<char, int> m; // char, the latest position of char
    int left = 0; // left boundary of substring without duplicate characters
    int sz = str.size();
    for (int i=0; i<sz; ++i) {
        if (m.count(str[i]) != 0) { // duplicate found
            left = max(left, m[str[i]]+1); // update left boundary
            m[str[i]] = i; // update occurrence of str[i] to the latest position
            ans = max(ans, i-left+1);
        } else {
            m[str[i]] = i; // update occurrence of str[i] to the latest position
            ans = max(ans, i-left+1);
        }
    }
    return ans;
}


void lengthOfLongestSubstring_scaffold(string input, int expectedResult) {
    Solution ss;
    int actual = ss.lengthOfLongestSubstring(input);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void computeArea_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    int actual = ss.computeArea(vi[0], vi[1], vi[2], vi[3], vi[4], vi[5], vi[6], vi[7]);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void isRectangleOverlap_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    vector<int> v2 = stringTo1DArray<int>(input2);
    int actual = ss.isRectangleOverlap(v1, v2);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


// 快速幂: https://zhuanlan.zhihu.com/p/95902286
int qpow2(int a, int n) {
    int ans = 1;
    while (n != 0) {
        if (n&1) {
            ans *= a;
        }
        a *= a;
        n >>= 1;
    }
    return ans;
}


int main() {
    SPDLOG_WARN("Running lengthOfLongestSubstring tests:");
    TIMER_START(lengthOfLongestSubstring);
    lengthOfLongestSubstring_scaffold("", 0);
    lengthOfLongestSubstring_scaffold("abba", 2);
    lengthOfLongestSubstring_scaffold("ababab", 2);
    lengthOfLongestSubstring_scaffold("bbbbb", 1);
    lengthOfLongestSubstring_scaffold("abcdef", 6);
    lengthOfLongestSubstring_scaffold("pwwkew", 3);
    lengthOfLongestSubstring_scaffold("dvdf", 3);
    lengthOfLongestSubstring_scaffold("aaabcdddd", 4);
    lengthOfLongestSubstring_scaffold("aaabcddadd", 4);
    TIMER_STOP(lengthOfLongestSubstring);
    SPDLOG_WARN("lengthOfLongestSubstring tests use {} ms", TIMER_MSEC(lengthOfLongestSubstring));

    SPDLOG_WARN("Running computeArea tests:");
    TIMER_START(computeArea);
    computeArea_scaffold("[-3,0,3,4,0,-1,9,2]", 45);
    computeArea_scaffold("[-2,-2,2,2,-2,-2,2,2]", 16);
    TIMER_STOP(computeArea);
    SPDLOG_WARN("computeArea tests use {} ms", TIMER_MSEC(computeArea));

    SPDLOG_WARN("Running isRectangleOverlap tests:");
    TIMER_START(isRectangleOverlap);
    isRectangleOverlap_scaffold("[-3,0,3,4]", "[0,-1,9,2]", 1);
    isRectangleOverlap_scaffold("[-2,-2,2,2]", "[-2,-2,2,2]", 1);
    isRectangleOverlap_scaffold("[-2,-2,2,2]", "[-2,-2,2,2]", 1);
    isRectangleOverlap_scaffold("[0,0,2,2]", "[1,1,3,3]", 1);
    isRectangleOverlap_scaffold("[0,0,1,1]", "[1,0,2,1]", 0);
    isRectangleOverlap_scaffold("[0,0,1,1]", "[2,2,3,3]", 0);
    TIMER_STOP(isRectangleOverlap);
    SPDLOG_WARN("isRectangleOverlap tests use {} ms", TIMER_MSEC(isRectangleOverlap));

    SPDLOG_WARN("Running rotate tests:");
    TIMER_START(rotate);
    rotate_scaffold("[1,2,3,4,5,6,7]", 3, "[5,6,7,1,2,3,4]");
    rotate_scaffold("[-1,-100,3,99]", 2, "[3,99,-1,-100]");
    TIMER_STOP(rotate);
    SPDLOG_WARN("rotate tests use {} ms", TIMER_MSEC(rotate));

    TIMER_START(merge);
    merge_scaffold("[[1,3],[2,6],[8,10],[15,18]]", "[[1,6],[8,10],[15,18]]");
    merge_scaffold("[[1,4],[4,5]]", "[[1,5]]");
    TIMER_STOP(merge);
    SPDLOG_WARN("merge tests use {} ms", TIMER_MSEC(merge));
}
