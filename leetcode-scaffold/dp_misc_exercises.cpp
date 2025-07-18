#include "leetcode.h"

using namespace std;

/* leetcode: 1139, 688, 576, 935, 322, 377  */
class Solution {
public:
    int largest1BorderedSquare(vector<vector<int>>& grid); // unsolved
    double knightProbability(int n, int k, int row, int column); // unsolved
    int knightDialer(int n); // unsolved
    int findPaths(int m, int n, int maxMove, int startRow, int startColumn);

};

int Solution::largest1BorderedSquare(vector<vector<int>>& grid) {
/*
    Given a 2D grid of 0s and 1s, return the number of elements in the largest *square* subgrid that has all 1s on its border, or 0 if such a subgrid doesn't exist in the grid.
*/


    return 0;
}

double knightProbability(int n, int k, int row, int column) {
/*
On an n x n chessboard, a knight starts at the cell (row, column) and attempts to make exactly k moves. The rows and columns are 0-indexed, so the top-left cell is (0, 0), and the bottom-right cell is (n - 1, n - 1).
A chess knight has eight possible moves it can make, as illustrated below. Each move is two cells in a cardinal direction, then one cell in an orthogonal direction.
Each time the knight is to move, it chooses one of eight possible moves uniformly at random (even if the piece would go off the chessboard) and moves there.
The knight continues moving until it has made exactly k moves or has moved off the chessboard.
Return the probability that the knight remains on the board after it has stopped moving.
*/
    return 0;
}

/*
There is an m x n grid with a ball. The ball is initially at the position [startRow, startColumn]. You are allowed to move the ball to one of the four adjacent cells in the grid (possibly out of the grid crossing the grid boundary). You can apply at most maxMove moves to the ball.

Given the five integers m, n, maxMove, startRow, startColumn, return the number of paths to move the ball out of the grid boundary. Since the answer can be very large, return it modulo 10^9 + 7.
*/
int Solution::findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
    const int mod = 10^9 + 7;
    // we may traverse a cell multiple times
    function<int(int, int, int)> dfs = [&] (int r, int c, int move) {
        if (move >= maxMove) { // termination
            return 0;
        }
        int count = 0;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (nr<0 || nr>=m || nc<0 || nc>=n) { // out of grid
                count = (count+1)%mod;
                continue;
            }
            count = (count + dfs(nr, nc, move+1))%mod;
        }
        return count;
    };
    return dfs(startRow, startColumn, 0);
}


int Solution::knightDialer(int n) {
/*
    Given an integer n, return how many distinct phone numbers of length n we can dial.
    You are allowed to place the knight on any numeric cell initially and then you should perform n - 1 jumps to dial a number of length n.
    All jumps should be valid knight jumps. As the answer may be very large, return the answer modulo 109 + 7.
*/
    return 0;
}


void largest1BorderedSquare_scaffold(string input, int expectedResult) {
    Solution ss;
    auto grid = stringTo2DArray<int>(input);
    int actual = ss.largest1BorderedSquare(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void findPaths_scaffold(string input, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input);
    int actual = ss.findPaths(vi[0], vi[1], vi[2], vi[3], vi[4]);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running largest1BorderedSquare tests:");
    TIMER_START(largest1BorderedSquare);
    largest1BorderedSquare_scaffold("[[1,1,1],[1,0,1],[1,1,1]]", 9);
    largest1BorderedSquare_scaffold("[[1,1],[0,0]]", 1);
    TIMER_STOP(largest1BorderedSquare);
    SPDLOG_WARN("largest1BorderedSquare tests use {} ms", TIMER_MSEC(largest1BorderedSquare));

    SPDLOG_WARN("Running findPaths tests:");
    TIMER_START(findPaths);
    findPaths_scaffold("[2,2,2,0,0]", 6);
    findPaths_scaffold("[1,3,3,0,1]", 12);
    //findPaths_scaffold("[1,2,50,0,0]", 150);
    TIMER_STOP(findPaths);
    SPDLOG_WARN("findPaths tests use {} ms", TIMER_MSEC(findPaths));
}
