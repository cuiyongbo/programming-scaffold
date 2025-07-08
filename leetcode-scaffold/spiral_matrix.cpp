#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix);
    // similar to <rotate array> problems
    void rotate(vector<vector<int>>& matrix);
    void setZeroes(vector<vector<int>>& matrix);

};



/*
Given an m x n matrix, return all elements of the matrix in spiral order.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
 
Constraints:
m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100
*/
vector<int> Solution::spiralOrder(vector<vector<int>>& matrix) {
/*
traversal order: right -> down -> left -> up

---->
^   |
|   |
|<--v
*/
    int m = matrix.size();
    int n = matrix[0].size();
    int up = 0;
    int down = m-1;
    int left = 0;
    int right = n-1;
    vector<int> ans;
    while (true) {
        // row: go right
        for (int i=left; i<=right; i++) {
            ans.push_back(matrix[up][i]);
        }
        up++;
        if (up > down) {
            break;
        }
        // column: go down
        for (int i=up; i<=down; i++) {
            ans.push_back(matrix[i][right]);
        }
        right--;
        if (left>right) {
            break;
        }
        // row: go left
        for (int i=right; i>=left; i--) {
            ans.push_back(matrix[down][i]);
        }
        down--;
        if (up > down) {
            break;
        }
        // column: go up
        for (int i=down; i>=up; i--) {
            ans.push_back(matrix[i][left]);
        }
        left++;
        if (left>right) {
            break;
        }
    }
    return ans;
}


void spiralOrder_scaffold(string input1, string expectedResult) {
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray<int>(input1);
    vector<int> actual = ss.spiralOrder(matrix);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input1, expectedResult, numberVectorToString(actual));
    }
}


/*
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
 
Constraints:
n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000
*/
void Solution::rotate(vector<vector<int>>& matrix) {
    // we need move matrix[i][j] to matrix[j][n-i-1]
    // 1. flip matrix upside down
    int n = matrix.size();
    for (int i=0; i<n/2; i++) {
        std::swap(matrix[i], matrix[n-i-1]);
    }
    // 2. transpose matrix alone the main diagnoal
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
}


void rotate_scaffold(string input1, string expectedResult) {
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray<int>(input1);
    ss.rotate(matrix);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (matrix == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input1, expectedResult);
        for (auto m: matrix) {
            print_vector(m);
        }
    }
}


/*
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's. You must do it in place.
 
Example 1:
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

Constraints:
m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-231 <= matrix[i][j] <= 231 - 1

Follow up:
A straightforward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
*/
void Solution::setZeroes(vector<vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    vector<bool> rows(m, false);
    vector<bool> columns(n, false);
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (matrix[i][j] == 0) {
                rows[i] = true;
                columns[j] = true;
            }
        }
    }
    for (int i=0; i<m; i++) {
        if (!rows[i]) {
            continue;
        }
        for (int j=0; j<m; j++) {
            matrix[i][j] = 0;
        }
    }
    for (int i=0; i<n; i++) {
        if (!columns[i]) {
            continue;
        }
        for (int j=0; j<n; j++) {
            matrix[j][i] = 0;
        }
    }
}


void setZeroes_scaffold(string input1, string expectedResult) {
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray<int>(input1);
    ss.setZeroes(matrix);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (matrix == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input1, expectedResult);
        for (auto m: matrix) {
            print_vector(m);
        }
    }
}


int main() {
    SPDLOG_WARN("Running spiralOrder tests: ");
    TIMER_START(spiralOrder);
    spiralOrder_scaffold("[[1,2,3],[4,5,6],[7,8,9]]", "[1,2,3,6,9,8,7,4,5]");
    spiralOrder_scaffold("[[1,2,3,4],[5,6,7,8],[9,10,11,12]]", "[1,2,3,4,8,12,11,10,9,5,6,7]");
    TIMER_STOP(spiralOrder);
    SPDLOG_WARN("spiralOrder using {} ms", TIMER_MSEC(spiralOrder));

    SPDLOG_WARN("Running rotate tests: ");
    TIMER_START(rotate);
    rotate_scaffold("[[1,2,3],[4,5,6],[7,8,9]]", "[[7,4,1],[8,5,2],[9,6,3]]");
    rotate_scaffold("[[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]", "[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]");
    TIMER_STOP(rotate);
    SPDLOG_WARN("rotate using {} ms", TIMER_MSEC(rotate));

    SPDLOG_WARN("Running setZeroes tests: ");
    TIMER_START(setZeroes);
    setZeroes_scaffold("[[1,2,3],[4,5,6],[7,8,9]]", "[[1,2,3],[4,5,6],[7,8,9]]");
    setZeroes_scaffold("[[1,1,1],[1,0,1],[1,1,1]]", "[[1,0,1],[0,0,0],[1,0,1]]");
    setZeroes_scaffold("[[0,1,2,0],[3,4,5,2],[1,3,1,5]]", "[[0,0,0,0],[0,4,5,0],[0,3,1,0]]");
    TIMER_STOP(setZeroes);
    SPDLOG_WARN("setZeroes using {} ms", TIMER_MSEC(setZeroes));

}

