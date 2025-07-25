#include "leetcode.h"

using namespace std;

/* leetcode: 150, 3, 223, 836, 189, 56 */
class Solution {
public:
    int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2);
    bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2);
};


/*
An axis-aligned rectangle is represented as a list [x1, y1, x2, y2], where (x1, y1) is the coordinate of its bottom-left corner, 
and (x2, y2) is the coordinate of its top-right corner. Its top and bottom edges are parallel to the X-axis, and its left and right edges are parallel to the Y-axis.
Two rectangles overlap if the area of their intersection is positive. To be clear, two rectangles that only touch at the corner or edges do not overlap.
Given two axis-aligned rectangles rec1 and rec2, return true if they overlap, otherwise return false.
*/
bool Solution::isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
    // vector<int>& rec1: x1, y1, x2, y2
    if (rec1[0] >= rec2[2] || rec2[0] >= rec1[2] || // separated from x-axis
        rec1[1] >= rec2[3] || rec2[1] >= rec1[3] ) { // separated from y-axis
        return false;
    } else {
        return true;
    }
}


/*
Given the coordinates of two rectilinear rectangles in a 2D plane, return the total area covered by the two rectangles. (union)
The first rectangle is defined by its bottom-left corner (ax1, ay1) and its top-right corner (ax2, ay2).
The second rectangle is defined by its bottom-left corner (bx1, by1) and its top-right corner (bx2, by2).
Hint: ans = area1 + area2 - intersection_area

rectangle q:
        (ax2,ay2)
__________
|        |           
|        |           
|        |           
----------
(ax1,ay1)

rectangle b:
        (bx2,by2)
__________
|        |           
|        |           
|        |           
----------
(bx1,by1)
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
        // max(bottom-left), min(upper-right)
        intersection = area(max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2));
    }
    return ans - intersection;
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

}
