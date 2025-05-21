#include "leetcode.h"

using namespace std;

class Solution {
public:
    int largestSubset(vector<int>& A);
    int missingNumber(vector<int> &arr);
    bool appointmentSlots(vector<int>& A, vector<int>& B, int S);
    int longestPath(vector<int>& parent, string s);
};


/*
problem: https://www.geeksforgeeks.org/find-the-size-of-largest-subset-with-positive-bitwise-and/

Given an array arr[] consisting of N positive integers, the task is to find the largest size of the subset of the array arr[] with positive Bitwise AND.

Examples:

Input: arr[] = [7, 13, 8, 2, 3]
Output: 3
Explanation:
The subsets having Bitwise AND positive are {7,13,3} and {7,2,3}  are of length 3, which is of maximum length among all possible subsets.

Input: arr[] = [1, 2, 4, 8]
Output: 1

Hint: 

Approach: The given problem can be solved by counting the number of set bits at each corresponding bits position for all array elements and then the count of the maximum of set bits at any position is the maximum count of subset because the Bitwise AND of all those elements is always positive

7 -->  00111
13 --> 01101
 8 --> 01000
 2 --> 00010
 3 --> 00011
       ------
       02233 <-- Evident BitWise AND bit(Most number of 1's in bit grid)

From above it is clearly evident that we can have maximum of 3 bitwise combinations 
where combinations are listed below as follows:         
{7,13,3}
{7,2,3}
*/
int Solution::largestSubset(vector<int>& A) {
    vector<int> bit_count(32, 0); // the input is of type int32
    for (auto n: A) {
        int x = 0;
        while (n > 0) {
            if (n & 1) {
                bit_count[x]++;
            }
            x++;
            n>>=1;
        }
    }
    auto p = max_element(bit_count.begin(), bit_count.end());
    return *p;
}


/*
// https://www.geeksforgeeks.org/find-the-smallest-positive-number-missing-from-an-unsorted-array/
Given an unsorted array arr[] with both positive and negative elements, the task is to find the smallest positive number missing from the array.

Examples:

Input: arr[] = {2, -3, 4, 1, 1, 7}
Output: 3
Explanation: 3 is the smallest positive number missing from the array.

Input: arr[] = {5, 3, 2, 5, 1}
Output: 4
Explanation: 4 is the smallest positive number missing from the array.

Input: arr[] = {-8, 0, -1, -4, -3}
Output: 1 
Explanation: 1 is the smallest positive number missing from the array.
*/
int Solution::missingNumber(vector<int> &A) {
    std::sort(A.begin(), A.end(), std::less<int>());
    int ans = 1;
    for (int i=0; i<(int)A.size(); i++) {
        if (A[i] == ans) {
            // ans is not missing, so increase ans to check next candidate
            ans++;
        } else if (A[i] > ans) {
            // current element A[i] is larger than ans, we are sure that ans is missing from A
            break;
        } else {
            // A[i] is negative. do nothing
        }
    }
    return ans;
}


/*
There are N patients (numbered from 0 to N-1) who want to visit the doctor. The doctor has S possible appointment slots, numbered from 1 to S. Each of the patients has two preferences. Patient K would like to visit the doctor during either slot A[K] or slot B[K]. The doctor can treat only one patient during each slot.

Is it possible to assign every patient to one of their preferred slots so that there will be at most one patient assigned to each slot?

that, given two arrays A and B, both of N integers, and an integer S, returns true if it is possible to assign every patient to one of their preferred slots, one patient to one slot, and false otherwise.

Examples:

Given A = [1, 1, 3], B = [2, 2, 1] and S = 3, the function should return true. We could assign patients in the following way: [1, 2, 3], where the K-th element of the array represents the number of the slot to which patient K was assigned. Another correct assignment would be [2, 1, 3]. On the other hand, [2, 2, 1] would be an incorrect assignment as two patients would be assigned to slot 2.
Given A = [3, 2, 3, 1], B = [1, 3, 1, 2] and S = 3, the function should return false. There are only three slots available, but there are four patients who want to visit the doctor. It is therefore not possible to assign the patients to the slots so that only one patient at a time would visit the doctor.
Given A = [2, 5, 6, 5], B = [5, 4, 2, 2] and S = 8, the function should return true. For example, we could assign patients in the following way: [5, 4, 6, 2].
Given A = [1, 2, 1, 6, 8, 7, 8], B = [2, 3, 4, 7, 7, 8, 7] and S = 10, the function should return false. It is not possible to assign all of the patients to one of their preferred slots so that only one patient will visit the doctor during one slot.

Constraints:
each element of array A and B is an integer within the range [1…S];
no patient has two preferences for the same slot, L.e. A[i] != B[i].
*/
bool Solution::appointmentSlots(vector<int>& A, vector<int>& B, int S) {
    int patients = A.size();
    vector<bool> visited(S, false);
    std::function<bool(int)> backtrace = [&] (int u) {
        if (u == patients) {
            return true;
        }
        if (!visited[A[u]]) {
            visited[A[u]] = true;
            if (backtrace(u+1)) {
                return true;
            }
            visited[A[u]] = false;
        }
        if (!visited[B[u]]) {
            visited[B[u]] = true;
            if (backtrace(u+1)) {
                return true;
            }
            visited[B[u]] = false;
        }
        return false;
    };
    return backtrace(0);
}


/*
You are given a tree (i.e. a connected, undirected graph that has no cycles) rooted at node 0 consisting of n nodes numbered from 0 to n-1. The tree is represented by a 0-indexed array parent of size n, where parent[i] is the parent of node i. Since node 0 is the root, parent[0] == -1.

You are also given a string s of length n, where s[i] is the character assigned to node i.

Return the length of the longest path in the tree such that no pair of adjacent nodes on the path have the same character assigned to them.

Examples:

Input: parent = [-1,0,0,1,1,2], s = "abacbe"
Output: 3
Explanation: The longest path where each two adjacent nodes have different characters in the tree is the path: 0 -> 1 -> 3. The length of this path is 3, so 3 is returned.
It can be proven that there is no longer path that satisfies the conditions.

Input: parent = [-1,0,0,0], s = "aabc"
Output: 3
Explanation: The longest path where each two adjacent nodes have different characters is the path: 2 -> 0 -> 3. The length of this path is 3, so 3 is returned.

Constraints:
    n == parent.length == s.length
    1 <= n <= 105
    0 <= parent[i] <= n-1 for all i >= 1
    parent[0] == -1
    parent represents a valid tree.
    s consists of only lowercase English letters.
*/
int Solution::longestPath(vector<int>& parent, string s) {
    // construct a directed graph
    int n = parent.size();
    vector<vector<int>> graph(n);
    for (int i=1; i<n; i++) {
        graph[parent[i]].push_back(i);
    }
    int ans = 0;
    vector<bool> visited(n, false);
    // return the maximum length of one parent-to-child path without node u, where no pair of adjacent nodes with the same letter
    function<int(int)> dfs = [&] (int u) {
        visited[u] = true;
        int max_leg = 0;
        for (auto v: graph[u]) {
            if (!visited[v]) {
                int x = dfs(v) + 1;
                if (s[u] != s[v]) {
                    ans = max(ans, max_leg+x);
                    max_leg = max(max_leg, x);
                }
            }
        }
        return max_leg;
    };
    dfs(0);
    return ans + 1;
}


void largestSubset_scaffold(string input1, int expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    int actual  = ss.largestSubset(v1);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input1, expectedResult, actual);
    }
}


void missingNumber_scaffold(string input1, int expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    int actual  = ss.missingNumber(v1);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input1, expectedResult, actual);
    }
}


void appointmentSlots_scaffold(string input1, string input2, int input3, bool expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    vector<int> v2 = stringTo1DArray<int>(input2);
    bool actual  = ss.appointmentSlots(v1, v2, input3);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input1, input2, input3, expectedResult, actual);
    }
}


void longestPath_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    int actual  = ss.longestPath(v1, input2);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running largestSubset tests:");
    TIMER_START(largestSubset);
    largestSubset_scaffold("[7, 13, 8, 2, 3]", 3);
    largestSubset_scaffold("[1,2,4,8]", 1);
    largestSubset_scaffold("[1,2,3,4,8]", 2);
    TIMER_STOP(largestSubset);
    SPDLOG_WARN("largestSubset tests use {} ms", TIMER_MSEC(largestSubset));

    SPDLOG_WARN("Running missingNumber tests:");
    TIMER_START(missingNumber);
    missingNumber_scaffold("[2, -3, 4, 1, 1, 7]", 3);
    missingNumber_scaffold("[5, 3, 2, 5, 1]", 4);
    missingNumber_scaffold("[-8, 0, -1, -4, -3]", 1);
    TIMER_STOP(missingNumber);
    SPDLOG_WARN("missingNumber tests use {} ms", TIMER_MSEC(missingNumber));

    SPDLOG_WARN("Running appointmentSlots tests:");
    TIMER_START(appointmentSlots);
    appointmentSlots_scaffold("[1, 1, 3]", "[2, 2, 1]", 3, true);
    appointmentSlots_scaffold("[3, 2, 3, 1]", "[1, 3, 1, 2]", 3, false);
    appointmentSlots_scaffold("[2, 5, 6, 5]", "[5, 4, 2, 2]", 8, true);
    appointmentSlots_scaffold("[1, 2, 1, 6, 8, 7, 8]", "[2, 3, 4, 7, 7, 8, 7]", 10, false);
    TIMER_STOP(appointmentSlots);
    SPDLOG_WARN("appointmentSlots tests use {} ms", TIMER_MSEC(appointmentSlots));

    SPDLOG_WARN("Running longestPath tests:");
    TIMER_START(longestPath);
    longestPath_scaffold("[-1,0,0,1,1,2]", "abacbe", 3);
    longestPath_scaffold("[-1,0,0,0]", "aabc", 3);
    TIMER_STOP(longestPath);
    SPDLOG_WARN("longestPath tests use {} ms", TIMER_MSEC(longestPath));
}
