#include "leetcode.h"

using namespace std;

/* leetcode: 11, 42 */
class Solution {
public:
    int maxArea(const vector<int>& height);
    int trap(vector<int>& height);
};


/*
Given n non-negative integers a1, a2, …, an, where each represents a point at coordinate (i, ai).
n vertical lines are drawn such that the two endpoints of line i is at (i, 0) and (i, ai).
Find two lines, which together with x-axis forms a container, such that the container contains the most water.
Note: You may not slant the container and n is at least 2.

For example, Given input: [1 3 2 4], output: 6
explanation: use 3, 4, we have the following, which contains (4th-2nd) * min(3, 4) = 2 * 3 = 6 unit of water.

            | 
    |       |
    |   |   |
|___|___|___|__
0   1   2   3  
*/
int Solution::maxArea(const vector<int>& height) {
    int ans = 0;
    int l = 0;
    int r = height.size() - 1; // r is inclusive
    while (l < r) {
        ans = std::max(ans, std::min(height[l], height[r])*(r-l));
        // move towards the direction that may maxify the answer
        // when height[l] is less than height[r], we know height[l] is used as height to calculate the possible ans.
        // then to maxify ans on the next move, we move l towards right, hoping that height[l+1] would be larger than height[l], 
        // which may result in a larger ans. the same is with the case when height[l] >= height[r] and we move r towards left.
        if (height[l]<height[r]) {
            l++;
        } else {
            r--;
        }
    }
    return ans;
}


/*
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it is able to trap after raining.
The following elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped:
              _        
      _      | |_   _  
  _  | |_   _| | |_| |_
_|_|_|_|_|_|_|_|_|_|_|_|
*/
int Solution::trap(vector<int>& height) {
    int ans = 0;
    int l=0, max_l = height[l];
    int r=height.size()-1, max_r = height[r]; // r is inclusive
    while (l < r) {
        //printf("l=%d, r=%d, max_l=%d, max_r=%d, ans=%d\n", l, r, max_l, max_r, ans);
        if (max_l < max_r) {
            // move l towards right
            ans += max_l - height[l];
            l++;
            max_l = std::max(max_l, height[l]); // update max_l
        } else {
            // move r towards left
            ans += max_r - height[r];
            r--;
            max_r = std::max(max_r, height[r]); // update max_l
        }
    }
    return ans;
}


void maxArea_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input);
    int actual = ss.maxArea(g);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void trap_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input);
    int actual = ss.trap(g);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

int main() {
    SPDLOG_WARN("Running maxArea tests:");
    TIMER_START(maxArea);
    maxArea_scaffold("[1,1]", 1);
    maxArea_scaffold("[1,3,2,4]", 6);
    maxArea_scaffold("[1,8,6,2,5,4,8,3,7]", 49);
    TIMER_STOP(maxArea);
    SPDLOG_WARN("maxArea tests use {} ms", TIMER_MSEC(maxArea));

    SPDLOG_WARN("Running trap tests:");
    TIMER_START(trap);
    trap_scaffold("[1,3,2,4]", 1);
    trap_scaffold("[0,1,0,2,1,0,1,3,2,1,2,1]", 6);
    trap_scaffold("[3,2,1,2,1,2,1,2,1]", 3);
    trap_scaffold("[3,2,0,2,0,2,0,2,1]", 6);
    TIMER_STOP(trap);
    SPDLOG_WARN("trap tests use {} ms", TIMER_MSEC(trap));
}
