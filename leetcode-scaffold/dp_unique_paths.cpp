#include "leetcode.h"

using namespace std;

/* leetcode: 62, 63, 64, 120, 174, 931, 1210, 1289 */
class Solution {
public:
    int uniquePaths(int m, int n);
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
    int minPathSum(vector<vector<int>>& grid);
    int minimumTotal(vector<vector<int>>& t);
    int minFallingSum_931(vector<vector<int>>& grid);
    int minFallingSum_1289(vector<vector<int>>& grid);
    int calculateMinimumHP(vector<vector<int>>& dungeon);
    int minimumMoves(vector<vector<int>>& grid);
};

/*
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). 
The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.
Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
*/
int Solution::uniquePaths(int m, int n) {
// dp[i][j] means the number of unique paths to (i, j)
// dp[i][j] = dp[i-1][j] + dp[i][j-1] // from upper or left
{ // naive solution
    vector<vector<int>> dp(m, vector<int>(n, 0));
    dp[0][0] = 1; // trivcal cases
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            if (i > 0) { // from upper
                dp[i][j] += dp[i-1][j];
            }
            if (j > 0) { // from left
                dp[i][j] += dp[i][j-1];
            }
        }
    }
    return dp[m-1][n-1];
}

{ // solution with optimization of space usage
    vector<vector<int>> dp(2, vector<int>(n, 0));
    dp[0][0] = 1; dp[1][0] = 1; // initialization
    for (int i=0; i<m; ++i) {
        // we only use the last row when calculation current row
        // and the other previous rows will never be used again
        for (int j=0; j<n; ++j) {
            if (i > 0) { // up
                dp[1][j] += dp[0][j];
            }
            if (j > 0) { // left
                dp[1][j] += dp[1][j-1];
            }
        }
        dp[0] = dp[1]; // update the last row
        dp[1].assign(n, 0);
    }
    return dp[0][n-1];
}

}


/*
Follow up for “Unique Paths”:
Now consider if some obstacles are added to the grids. How many unique paths would there be?
An obstacle and empty space are marked as 1 and 0 respectively in the grid.
*/
int Solution::uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    // dp[i][j] means the number of unique paths to (i, j)
    // dp[i][j] += dp[i-1][j] if obstacleGrid[i-1][j]==0 (upper)
    // dp[i][j] += dp[i][j-1] if obstacleGrid[i][j-1]==0 (left)
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    dp[0][0] = 1; // initialization
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            if (i>0 && obstacleGrid[i-1][j]==0) { // from upper
                dp[i][j] += dp[i-1][j];
            }
            if (j>0 && obstacleGrid[i][j-1]==0) { // from left
                dp[i][j] += dp[i][j-1];
            }
        }
    }
    return dp[m-1][n-1];
}


/*
Given a mxn grid filled with non-negative numbers, find a path from top-left to bottom-right which minimizes the sum of all numbers along its path.
Note: You can only move either down or right at any point in time.
Example 1:
    [[1,3,1],
    [1,5,1],
    [4,2,1]]
Given the above grid map, return 7. Because the path 1→3→1→1→1 minimizes the sum.
*/
int Solution::minPathSum(vector<vector<int>>& grid) {
// dp[i][j] means minPathSum to (i, j)
// dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1]);
if (0) { // dp solution
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            int cost = INT32_MAX;
            if (i > 0) { // from upper
                cost = min(cost, dp[i-1][j]);
            }
            if (j > 0) { // from left
                cost = min(cost, dp[i][j-1]);
            }
            dp[i][j] = (cost == INT32_MAX) ? grid[i][j] : (cost+grid[i][j]);
        }
    }
    return dp[m-1][n-1];
}

{ // solution using dijkstra algorithm
    using Coordinate = std::pair<int, int>; // row, column
    using element_type = std::pair<Coordinate, int>; // Coordinate, path_cost
    int rows = grid.size();
    int columns = grid[0].size();
    Coordinate start = make_pair(0, 0);
    Coordinate end = make_pair(rows-1, columns-1);
    vector<Coordinate> directions {
        {1, 0}, // go down
        {0, 1}, // go right
    };
    std::map<Coordinate, int> visited; // coor, min_path_cost from start to coor
    visited[start] = grid[0][0]; // why not set visited[start] to 0? because the value means the sum of the numbers on the path ending with start
    auto cmp = [&] (const element_type& l, const element_type& r) {
        return l.second > r.second;
    };
    std::priority_queue<element_type, vector<element_type>, decltype(cmp)> pq(cmp); // min-heap
    pq.emplace(start, grid[0][0]);
    while (!pq.empty()) {
        auto t = pq.top(); pq.pop();
        //printf("(%d,%d):%d\n", t.first.first, t.first.second, t.second);
        if (t.first == end) {
            return t.second;
        }
        for (auto& d: directions) {
            int nr = t.first.first + d.first;
            int nc = t.first.second + d.second;
            if (nr<0 || nr>=rows || nc<0 || nc>=columns) {
                continue;
            }
            auto p = make_pair(nr, nc);
            if (visited.count(p)==0 /*not visited*/
                || (visited[p]>t.second+grid[nr][nc]) /*find a better path*/
            ) {
                visited[p] = t.second+grid[nr][nc];
                pq.emplace(p, visited[p]);
            }
        }
    }
    return -1;
}

}


/*
Given a triangle array, return the minimum path sum from top to bottom. For each step, you may move to an adjacent number of the row below. 
More formally, if you are on index i on the current row, you may move to either index i or index i+1 on the next row.
For example, given the following triangle: 
(r, c) -> (r+1, c), (r+1, c+1)
(r-1, c-1), (r-1, c) -> (r, c)
[
       [2],
      [3,4],
     [6,5,7],
    [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
*/
int Solution::minimumTotal(vector<vector<int>>& t) {
// dp[i][j] means minimumTotal to reach (i, j)
// dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]) + t[i][j]
{
    int ans = INT32_MAX;
    int m = t.size();
    vector<vector<int>> dp(m, vector<int>(m, 0));
    for (int i=0; i<m; ++i) {
        for (int j=0; j<=i; ++j) {
            // Note that there are i element(s) in the (i-1)-th row
            int last_row_size = i;
            int cost = INT32_MAX;
            if (i>0 && j>0 && j-1<=last_row_size-1) { // (j-1<=i-1) ensure we won't be out of range when visiting previous row
                cost = min(cost, dp[i-1][j-1]);
            }
            if (i>0 && j<=last_row_size-1) { // (j<=i-1) ensure we won't be out of range when visiting previous row
                cost = min(cost, dp[i-1][j]);
            }
            dp[i][j] = cost==INT32_MAX ? t[i][j] : cost+t[i][j];
            if (i == m-1) {
                ans = min(ans, dp[i][j]);
            }
        }
    }
    return ans;
}
}


/*
Given an nxn array of integers matrix, return the minimum sum of any falling path through matrix.

A falling path starts at any element in the first row and chooses the element in the next row that 
is either directly below or diagonally left/right. Specifically, the next element from position (row, col) 
will be (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).

as illustrated by the diagram:

(r-1,c-1)(r-1,c)(r-1,c+1)
           \|/
   (r,c-1)(r,c)(r,c+1)
           /|\
(r+1,c-1)(r+1,c)(r+1,c+1)

Example 1:
    Input: [[1,2,3],[4,5,6],[7,8,9]]
    Output: 12
    The possible falling paths are:
        [1,4,7], [1,4,8], [1,5,7], [1,5,8], [1,5,9]
        [2,4,7], [2,4,8], [2,5,7], [2,5,8], [2,5,9], [2,6,8], [2,6,9]
        [3,5,7], [3,5,8], [3,5,9], [3,6,8], [3,6,9]
    The falling path with the smallest sum is [1,4,7], so the answer is 12.
*/
int Solution::minFallingSum_931(vector<vector<int>>& grid) {
// dp[i][j] means minFallingSum to reach (i, j)
// dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) + grid[i][j]
    int ans = INT32_MAX;
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<int>> dp(rows, vector<int>(columns, INT32_MAX));
    dp[0] = grid[0]; // initialization. for rows==1
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            int cost = INT32_MAX;
            if (r>0) {
                cost = min(cost, dp[r-1][c]); // from upper
                if (c>0) {
                    cost = min(cost, dp[r-1][c-1]); // from diagonally left
                }
                if (c+1<columns) {
                    cost = min(cost, dp[r-1][c+1]); // from diagonally right
                }
            } else {
                cost = 0;
            }
            dp[r][c] = cost + grid[r][c];
            // it would be more efficient to check ans here
            if (r == rows-1) {
                ans = min(ans, dp[r][c]);
            }
        }
    }
    return ans;
}


/*
Given an n x n integer matrix grid, return the minimum sum of a falling path with non-zero shifts.
A falling path with non-zero shifts is a choice of exactly one element from each row of grid such 
that no two elements chosen in adjacent rows are in the same column.
Example:
    Input: arr = [[1,2,3],[4,5,6],[7,8,9]]
    Output: 13
    Explanation: The possible falling paths are:
        [1,5,9], [1,5,7], [1,6,7], [1,6,8],
        [2,4,8], [2,4,9], [2,6,7], [2,6,8],
        [3,4,8], [3,4,9], [3,5,7], [3,5,9]
    The falling path with the smallest sum is [1,5,7], so the answer is 13.
*/
int Solution::minFallingSum_1289(vector<vector<int>>& grid) {
    // dp[i][j] means minFallingSum to reach (i, j)
    // dp[i][j] = min{dp[i-1][k]} + grid[i][j] for k in [0, n] if k!=j
    int ans = INT32_MAX;
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<int>> dp(rows, vector<int>(columns, INT32_MAX));
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            int cost = INT32_MAX;
            if (r > 0) {
                for (int k=0; k<columns; ++k) {
                    if (k != c) {
                        cost = min(cost, dp[r-1][k]);
                    }
                }
            } else { // for the first row
                cost = 0;
            }
            dp[r][c] = cost + grid[r][c];
            if (r == rows-1) {
                ans = min(ans, dp[r][c]);
            }
        }
    }
    return ans;
}


int Solution::minimumMoves(vector<vector<int>>& grid) {
/*
In an n*n grid, there is a snake that spans 2 cells and starts moving from the top-left corner at (0, 0) and (0, 1).
The grid has empty cells represented by 0 and blocked cells represented by 1.
The snake wants to reach the bottom-right corner at (n-1, n-2) and (n-1, n-1).

In one move the snake can:

    Move one cell to the right if there are no blocked cells there. 
    This move keeps the horizontal/vertical position of the snake as it is.

    Move down one cell if there are no blocked cells there. 
    This move keeps the horizontal/vertical position of the snake as it is.
    
    Rotate clockwise if it’s in a horizontal position and the two cells 
    under it are both empty. In that case the snake moves from (r, c) 
    and (r, c+1) to (r, c) and (r+1, c).
    
    Rotate counterclockwise if it’s in a vertical position and the two cells 
    to its right are both empty. In that case the snake moves from (r, c) 
    and (r+1, c) to (r, c) and (r, c+1).
    
Return the minimum number of moves to reach the target.
If there is no way to reach the target, return -1.

Example:

    Input: 
    grid = [[0,0,0,0,0,1],
            [1,1,0,0,1,0],
            [0,0,0,0,1,1],
            [0,0,1,0,1,0],
            [0,1,1,0,0,0],
            [0,1,1,0,0,0]]
    Output: 11
    Explanation:
    One possible solution is [right, right, rotate clockwise, right, down, 
    down, down, down, rotate counterclockwise, right, down].

    Example 2:
    Input: grid = [[0,0,1,1,1,1],
                    [0,0,0,0,1,1],
                    [1,1,0,0,0,1],
                    [1,1,1,0,0,1],
                    [1,1,1,0,0,1],
                    [1,1,1,0,0,0]]
    Output: 9

    Constraints:

        2 <= n <= 100
        0 <= grid[i][j] <= 1
        It is guaranteed that the snake starts at empty cells.
*/

    // used[i][j] = 0: not used
    // used[i][j] = 1: tail facing right used
    // used[i][j] = 2: tail facing down used
    // used[i][j] = 3: done

    int n  = (int)grid.size();
    vector<vector<int>> used(n, vector<int>(n, 0));
    auto is_right_used = [&](int y, int x) { return (used[y][x] & 0x1) != 0;};
    auto is_down_used = [&](int y, int x) { return (used[y][x] & 0x2) != 0;};
    auto set_right = [&](int y, int x) { used[y][x] |= 0x1;};
    auto set_down = [&](int y, int x) { used[y][x] |= 0x2;};

    auto is_target = [&] (const vector<int>& t) {
        return t[0] == n-1 && t[1] == n-2 && t[2] == n-1 && t[3] == n-1;
    };

    // perform bfs to find the minimum steps to reach the destination
    int steps = 0;
    set_right(0, 1);
    queue<vector<int>> q;
    q.push({0, 0, 0, 1}); // y1, x1, y2, x2
    while (!q.empty()) {
        for (size_t k=q.size(); k != 0; k--) {
            auto t = q.front(); q.pop();
            // reach the destination
            if (is_target(t)) {
                return steps;
            }
            int y = t[2], x = t[3];

            // horizontal
            if (t[0] == t[2]) {
                // go right
                if (x+1 < n && !grid[y][x+1] && !is_right_used(y, x+1)) {
                    set_right(y, x+1);
                    q.push({y, x, y, x+1});
                }

                if (y+1 < n && !grid[y+1][x-1] && !grid[y+1][x]) {
                    // go down
                    if (!is_right_used(y+1, x)) {
                        set_right(y+1, x);
                        q.push({y+1, x-1, y+1, x});
                    }

                    // rotate clockwise
                    if (!is_down_used(y+1, x-1)) {
                        set_down(y+1, x-1);
                        q.push({y, x-1, y+1, x-1});
                    }
                }
            }

            // vertical
            if (t[1] == t[3]) {
                // go down
                if (y+1 < n && !grid[y+1][x] && !is_down_used(y+1, x)) {
                    set_down(y+1, x);
                    q.push({y, x, y+1, x});
                }

                if (x+1 < n && !grid[y-1][x+1] && !grid[y][x+1]) {
                    // go right
                    if (!is_down_used(y, x+1)) {
                        set_down(y, x+1);
                        q.push({y-1, x+1, y, x+1});
                    }

                    // rotate counter-clockwise
                    if (!is_right_used(y-1, x+1)) {
                        set_right(y-1, x+1);
                        q.push({y-1, x, y-1, x+1});
                    }
                }
            }
        }
        ++steps;
    }
    return -1;
}


/*
The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms 
laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.
Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0s) or contain magic orbs that increase the knight’s health (positive integers).

In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

Write a function to determine the knight’s minimum initial health so that he is able to rescue the princess.

For example, given the dungeon below, the initial health of the knight must be at least 7 
if he follows the optimal path RIGHT-> RIGHT -> DOWN -> DOWN:
    [
        [-2 (K),-3,3],
        [-5,-10,1],
        [10,30,-5 (P)],
    ]
*/
int Solution::calculateMinimumHP(vector<vector<int>>& dungeon) {
// dp[i][j] means MinimumHP to get (m-1, n-1) from (i, j), assume (i, j) is the start point.
// dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1])-dungeon[i][j]))
// we can build the answer from bottom to top: what is the MinimumHP to go through the sub-dungeon (i, j) to (m-1, n-1), then extend the solution from (m-1, n-1) to (0, 0)
{ // naive dp solution, brilliant
    int m = dungeon.size();
    int n = dungeon[0].size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, INT32_MAX)); // we have to initialize dp with INT32_MAX
    // source: (0, 0)
    // destination: (m-1, n-1)
    // initialization: the knight starts from destination, and he must have 1 HP at least
    dp[m][n-1] = 1; // from downside
    dp[m-1][n] = 1; // from right
    for (int i=m-1; i>=0; --i) {
        for (int j=n-1; j>=0; --j) {
            dp[i][j] = max(1, // HP must be 1 at least
                            min(dp[i+1][j], dp[i][j+1])-dungeon[i][j]
                        );
        }
    }
    return dp[0][0];
}

{ // solution with optimization of space usage
    int m = dungeon.size();
    int n = dungeon[0].size();
    vector<int> dp(n+1, INT32_MAX); dp[n-1] = 1;
    for (int i=m-1; i>=0; i--) {
        for (int j=n-1; j>=0; j--) {
            dp[j] = max(1, min(dp[j], dp[j+1]))-dungeon[i][j];
        }
    }
    return dp[0];
}

}


void uniquePaths_scaffold(int input1, int input2, int expectedResult) {
    Solution ss;
    int actual = ss.uniquePaths(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


void uniquePathsWithObstacles_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.uniquePathsWithObstacles(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void minPathSum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minPathSum(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void minimumTotal_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minimumTotal(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void calculateMinimumHP_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.calculateMinimumHP(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void minFallingSum_scaffold(string input, int expectedResult, int func_no) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = 0;
    if (func_no == 931) {
        actual = ss.minFallingSum_931(grid);
    } else if (func_no == 1289){
        actual = ss.minFallingSum_1289(grid);
    } else {
        SPDLOG_ERROR("func_no can only be values in [931, 1289], actual={}", func_no);
        return;
    }
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}, func_no={}) passed", input, expectedResult, func_no);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}, func_no={}) failed, actual={}", input, expectedResult, func_no, actual);
    }
}

void minimumMoves_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minimumMoves(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

int main() {
    SPDLOG_WARN("Running uniquePaths tests:");
    TIMER_START(uniquePaths);
    uniquePaths_scaffold(3, 7, 28);
    uniquePaths_scaffold(3, 2, 3);
    TIMER_STOP(uniquePaths);
    SPDLOG_WARN("uniquePaths tests use {} ms", TIMER_MSEC(uniquePaths));

    SPDLOG_WARN("Running uniquePathsWithObstacles tests:");
    TIMER_START(uniquePathsWithObstacles);
    uniquePathsWithObstacles_scaffold("[[0,0,0],[0,1,0],[0,0,0]]", 2);
    uniquePathsWithObstacles_scaffold("[[0,1],[0,0]]", 1);
    TIMER_STOP(uniquePathsWithObstacles);
    SPDLOG_WARN("uniquePathsWithObstacles tests use {} ms", TIMER_MSEC(uniquePathsWithObstacles));

    SPDLOG_WARN("Running minPathSum tests:");
    TIMER_START(minPathSum);
    minPathSum_scaffold("[[1,3,1],[1,5,1],[4,2,1]]", 7);
    minPathSum_scaffold("[[1,2,3],[4,5,6]]", 12);
    TIMER_STOP(minPathSum);
    SPDLOG_WARN("minPathSum tests use {} ms", TIMER_MSEC(minPathSum));

    SPDLOG_WARN("Running minimumTotal tests:");
    TIMER_START(minimumTotal);
    minimumTotal_scaffold("[[2],[3,4],[6,5,7],[4,1,8,3]]", 11);
    minimumTotal_scaffold("[[-10]]", -10);
    minimumTotal_scaffold("[[1],[2,3],[4,5,6],[7,8,9,10]]", 14);
    TIMER_STOP(minimumTotal);
    SPDLOG_WARN("minimumTotal tests use {} ms", TIMER_MSEC(minimumTotal));

    SPDLOG_WARN("Running calculateMinimumHP tests:");
    TIMER_START(calculateMinimumHP);
    calculateMinimumHP_scaffold("[[-2,-3,3],[-5,-10,1],[10,30,-5]]", 7);
    calculateMinimumHP_scaffold("[[0]]", 1);
    calculateMinimumHP_scaffold("[[100]]", 1);
    calculateMinimumHP_scaffold("[[5,-3,2],[-2,4,-1],[3,-5,6]]", 1);
    TIMER_STOP(calculateMinimumHP);
    SPDLOG_WARN("calculateMinimumHP tests use {} ms", TIMER_MSEC(calculateMinimumHP));

    SPDLOG_WARN("Running minFallingSum tests:");
    TIMER_START(minFallingSum);
    minFallingSum_scaffold("[[7]]", 7, 931);
    minFallingSum_scaffold("[[1,2,3],[4,5,6],[7,8,9]]", 12, 931);
    minFallingSum_scaffold("[[-19,57],[-40,-5]]", -59, 931);
    minFallingSum_scaffold("[[7]]", 7, 1289);
    minFallingSum_scaffold("[[1,2,3],[4,5,6],[7,8,9]]", 13, 1289);
    minFallingSum_scaffold("[[-73,61,43,-48,-36],[3,30,27,57,10],[96,-76,84,59,-15],[5,-49,76,31,-7],[97,91,61,-46,67]]", -192, 1289);
    TIMER_STOP(minFallingSum);
    SPDLOG_WARN("minFallingSum tests use {} ms", TIMER_MSEC(minFallingSum));

    SPDLOG_WARN("Running minimumMoves tests:");
    TIMER_START(minimumMoves);
    minimumMoves_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 11);
    minimumMoves_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 9);
    TIMER_STOP(minimumMoves);
    SPDLOG_WARN("minimumMoves tests use {} ms", TIMER_MSEC(minimumMoves));
}
