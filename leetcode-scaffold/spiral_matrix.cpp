#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix);
    // similar to <rotate array> problems
    void rotate(vector<vector<int>>& matrix);
    void setZeroes(vector<vector<int>>& matrix);
    void gameOfLife(vector<vector<int>>& board);

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
    for (int j=0; j<n; j++) {
        if (!columns[j]) {
            continue;
        }
        for (int i=0; i<m; i++) {
            matrix[i][j] = 0;
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


/*
According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

The board is made up of an m x n grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

- Any live cell with fewer than two live neighbors dies as if caused by under-population.
- Any live cell with two or three live neighbors lives on to the next generation.
- Any live cell with more than three live neighbors dies, as if by over-population.
- Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the m x n grid board, return the next state.

Example 1:
Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]

Example 2:
Input: board = [[1,1],[1,0]]
Output: [[1,1],[1,1]]

Constraints:
m == board.length
n == board[i].length
1 <= m, n <= 25
board[i][j] is 0 or 1.
 
Follow up:

Could you solve it in-place? Remember that the board needs to be updated simultaneously: You cannot update some cells first and then use their updated values to update other cells.
In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches upon the border of the array (i.e., live cells reach the border). How would you address these problems?

Hint: we should distinguish the next state of current cell from current state if its state changed.
such as when a cell goes from live to dead, we mark its state to 2.(any number which is not 0 or 1). similarly when a cell goes from dead to live, we mark its state to -1(any number which is not 0 or 1, of course, not 2 either).
*/
void Solution::gameOfLife(vector<vector<int>>& board) {
    int m = board.size();
    int n = board[0].size();
    auto live_cell_num = [&] (int i, int j) {
        int num = 0;
        int nr = 0, nc = 0;
        // upper
        nr = i-1; nc = j;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // upper-right
        nr = i-1; nc = j+1;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // right
        nr = i; nc = j+1;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // bottom-right
        nr = i+1; nc = j+1;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // bottom
        nr = i+1; nc = j;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // bottom-left
        nr = i+1; nc = j-1;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // left
        nr = i; nc = j-1;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        // upper-left
        nr = i-1; nc = j-1;
        if (0<=nr&&nr<m&&0<=nc&&nc<n) {
            num += (board[nr][nc] >= 1);
        }
        return num;
    };
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            int cells = live_cell_num(i, j);
            if (cells < 2) {
                if (board[i][j] == 1) {
                    board[i][j] = 2; // live -> dead
                }
            } else if (cells == 2) {
                // do nothing
            } else if (cells == 3) {
                if (board[i][j] == 0) {
                    board[i][j] = -1; // dead -> live
                }
            } else {
                if (board[i][j] == 1) {
                    board[i][j] = 2; // live -> dead
                }
            }
        }
    }
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (board[i][j] == -1) {
                board[i][j] = 1;
            } else if (board[i][j] == 2) {
                board[i][j] = 0;
            }
        }
    }
}


void gameOfLife_scaffold(string input1, string expectedResult) {
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray<int>(input1);
    ss.gameOfLife(matrix);
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

    SPDLOG_WARN("Running gameOfLife tests: ");
    TIMER_START(gameOfLife);
    gameOfLife_scaffold("[[1,1,1],[1,0,1],[1,1,1]]", "[[1,0,1],[0,0,0],[1,0,1]]");
    gameOfLife_scaffold("[[0,1,0],[0,0,1],[1,1,1],[0,0,0]]", "[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]");
    gameOfLife_scaffold("[[1,1],[1,0]]", "[[1,1],[1,1]]");
    TIMER_STOP(gameOfLife);
    SPDLOG_WARN("gameOfLife using {} ms", TIMER_MSEC(gameOfLife));
}
