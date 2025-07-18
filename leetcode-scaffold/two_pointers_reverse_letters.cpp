#include "leetcode.h"

using namespace std;

/* leetcode: 855, 917, 925, 986 */
typedef vector<int> Interval;
class Solution {
public:
    string reverseOnlyLetters(string input);
    bool isLongPressedName(string name, string typed);
    vector<Interval> intervalIntersection(vector<Interval>& A, vector<Interval>& B);
};


/*
Given a string s, reverse the string according to the following rules:
    All the characters that are not English letters remain in the same position.
    All the English letters (lowercase or uppercase) should be reversed.
Return s after reversing it.

Example 1:
    Input: s = "ab-cd"
    Output: "dc-ba"
*/
string Solution::reverseOnlyLetters(string input) {
    auto is_letter = [](char c) { return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z'); };
    int sz = input.size();
    for (int i=0, j=sz-1; i<j;) {
        if (is_letter(input[i]) && is_letter(input[j])) {
            swap(input[i], input[j]); // perform swap only when both positions are characters 
            ++i; --j;
        } else if (!is_letter(input[i])) {
            ++i;
        } else if (!is_letter(input[j])) {
            --j;
        }
    }
    return input;
}


/*
Your friend is typing his name into a keyboard. Sometimes, when typing a character c, the key might get long pressed, and the character will be typed one or more times.
You examine the typed characters of the keyboard. Return True if it is possible that it was your friend's name, with some characters (possibly none) being long pressed. 
Note: the characters of name and typed are lowercase letters.
*/
bool Solution::isLongPressedName(string name, string typed) {
    int len1 = name.size();
    int len2 = typed.size();
    int i=0, j=0; 
    while (i<len1&&j<len2) {
        if (name[i] != typed[j]) {
            return false;
        }
        int c = name[i];
        // how many c in name?
        int p1 = i;
        while (i<len1 && name[i]==c) {
            ++i;
        }
        // how many c in typed?
        int p2 = j;
        while (j<len2 && typed[j]==c) {
            ++j;
        }
        // we may type certain character the same times as or more times than its occurrences in original `name` but NOT LESS
        if (i-p1 > j-p2) {
            return false;
        }
    }
    // no letter left for both name and typed
    return i==len1 && j==len2;
}


/*
Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order. Return the intersection of these two interval lists.

(Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  
The intersection of two closed intervals is a set of real numbers that is either empty, or can be 
represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)

Example 1:
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
*/
vector<Interval> Solution::intervalIntersection(vector<Interval>& A, vector<Interval>& B) {
    int len1 = A.size();
    int len2 = B.size();
    int i=0, j=0;
    vector<Interval> ans;
    while (i<len1 && j<len2) {
        if (A[i][0] > B[j][1]) { // rangeB | rangeA, no intersection
            ++j;
        } else if (B[j][0] > A[i][1]) { // rangeA | rangeB, no intersection
            ++i;
        } else { // the intersection between rangeA and rangeB is not empty
            ans.push_back({max(A[i][0], B[j][0]), min(A[i][1], B[j][1])});
            // move interval with smaller region forward
            if (A[i][1] < B[j][1]) {
                ++i;
            } else {
                ++j;
            }
        }
    }
    return ans;
}


void reverseOnlyLetters_scaffold(string input, string expectedResult) {
    Solution ss;
    string actual = ss.reverseOnlyLetters(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void isLongPressedName_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    bool actual = ss.isLongPressedName(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}

void intervalIntersection_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> A = stringTo2DArray<int>(input1);
    vector<vector<int>> B = stringTo2DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.intervalIntersection(A, B);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        for(const auto& s: actual) {
            cout << numberVectorToString(s) << endl;
        } 
    }
}


int main() {
    SPDLOG_WARN("Running reverseOnlyLetters tests:");
    TIMER_START(reverseOnlyLetters);
    reverseOnlyLetters_scaffold("ab-cd", "dc-ba");
    reverseOnlyLetters_scaffold("a-bC-dEf-ghIj", "j-Ih-gfE-dCba");
    reverseOnlyLetters_scaffold("Test1ng-Leet=code-Q!", "Qedo1ct-eeLg=ntse-T!");
    reverseOnlyLetters_scaffold("1a2", "1a2");
    reverseOnlyLetters_scaffold("123ab", "123ba");
    TIMER_STOP(reverseOnlyLetters);
    SPDLOG_WARN("reverseOnlyLetters tests use {} ms", TIMER_MSEC(reverseOnlyLetters));

    SPDLOG_WARN("Running isLongPressedName tests:");
    TIMER_START(isLongPressedName);
    isLongPressedName_scaffold("cherry", "cherry", true);
    isLongPressedName_scaffold("leelee", "lleeelee", true);
    isLongPressedName_scaffold("leelee", "lleeeel", false);
    isLongPressedName_scaffold("saeed", "ssaaeed", true);
    isLongPressedName_scaffold("alex", "aaleex", true);
    isLongPressedName_scaffold("alex", "alexd", false);
    isLongPressedName_scaffold("tlex", "alex", false);
    TIMER_STOP(isLongPressedName);
    SPDLOG_WARN("isLongPressedName tests use {} ms", TIMER_MSEC(isLongPressedName));

    SPDLOG_WARN("Running intervalIntersection tests:");
    TIMER_START(intervalIntersection);
    intervalIntersection_scaffold("[[0,2],[5,10],[13,23],[24,25]]", 
                                    "[[1,5],[8,12],[15,24],[25,26]]", 
                                    "[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]");
    intervalIntersection_scaffold("[[0,2],[5,10],[13,23],[24,25]]", 
                                    "[[0,2],[5,10],[13,23],[24,25]]", 
                                    "[[0,2],[5,10],[13,23],[24,25]]");
    TIMER_STOP(intervalIntersection);
    SPDLOG_WARN("intervalIntersection tests use {} ms", TIMER_MSEC(intervalIntersection));
}
