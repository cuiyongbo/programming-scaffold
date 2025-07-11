#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    bool isPalindrome(string s);
    vector<string> fullJustify(vector<string>& words, int maxWidth);
    // <two sum, three sum, three sum closest> problems
    // two_pointers_two_sum.cpp
    bool canConstruct(string ransomNote, string magazine);
    bool isIsomorphic(string s, string t);

};


/*
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:
Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.

Constraints:
1 <= s.length <= 2 * 105
s consists only of printable ASCII characters.
*/
bool Solution::isPalindrome(string s) {
    if (s.empty()) { // trivial case
        return true;
    }
    int left = 0;
    int right = s.size() - 1;
    while (left < right) {
        if (!std::isalnum(s[left])) {
            left++;
        } else if (!std::isalnum(s[right])) {
            right--;
        } else if (std::tolower(s[left]) == std::tolower(s[right])) {
            left++; right--;
        } else {
            return false;
        }
    }
    return true;
}


void isPalindrome_scaffold(string input1, int expectedResult) {
    Solution ss;
    bool actual = ss.isPalindrome(input1);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input1, expectedResult, actual);
    }
}


/*
Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified, and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
 

Example 1:
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

Example 2:
Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be", because the last line must be left-justified instead of fully-justified.
Note that the second line is also left-justified because it contains only one word.
Example 3:

Input: words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]
 
Constraints:

1 <= words.length <= 300
1 <= words[i].length <= 20
words[i] consists of only English letters and symbols.
1 <= maxWidth <= 100
words[i].length <= maxWidth
Solutions
*/
vector<string> Solution::fullJustify(vector<string>& words, int maxWidth) {
    vector<string> ans;
    int left = -1;
    int right = -1;
    int current_len = 0;
    int i = 0;
    int word_cnt = words.size();
    while (i < word_cnt) {
        int s = words[i].size();
        if (left == -1) {
            current_len += s;
            left = right = i;
            i++;
        } else {
            int space_cnt = right - left + 1;
            if (current_len+space_cnt+s == maxWidth) {
                // " ".join(words[left:right+1])
                string buffer;
                for (int j=left; j<=right; j++) {
                    buffer.append(words[j]);
                    buffer.push_back(' ');
                }
                buffer.append(words[i]);
                ans.push_back(buffer);
                // reset cursors
                current_len = 0;
                left = right = -1;
                i++;
            } else if (current_len+space_cnt+s > maxWidth) {
                // words[left:right]
                int total_space_num = maxWidth - current_len;
                int slots = right - left;
                if (slots == 0) {
                    string buffer;
                    buffer.append(words[right]);
                    buffer.append(maxWidth-buffer.size(), ' ');
                    ans.push_back(buffer);
                } else {
                    int avg = total_space_num / slots;
                    int remaining = total_space_num - slots * avg;
                    string buffer;
                    for (int j=left; j<right; j++) {
                        buffer.append(words[j]);
                        int spaces = avg;
                        if (remaining > 0) {
                            remaining--;
                            spaces++;
                        }
                        buffer.append(spaces, ' ');
                    }
                    buffer.append(words[right]);
                    if (buffer.size() < maxWidth) {
                        buffer.append(maxWidth-buffer.size(), ' ');
                    }
                    ans.push_back(buffer);
                }
                // reset cursors
                current_len = 0;
                left = right = -1;
                // DON't increase i here
            } else {
                current_len += s;
                right = i;
                i++;
            }
        }
    }
    // for the last line
    if (left != -1) {
        string buffer;
        for (int j=left; j<right; j++) {
            buffer.append(words[j]);
            buffer.push_back(' ');
        }
        buffer.append(words[right]);
        buffer.append(maxWidth-buffer.size(), ' '); // append whitespaces if necessary
        ans.push_back(buffer);
    }
    // DEBUG
    //for (const auto& s: ans) {
    //    printf("maxWidth: %d, len: %d, [%s]\n", maxWidth, (int)s.size(), s.c_str());
    //}
    return ans;
}


void fullJustify_scaffold(string input1, int input2) {
    Solution ss;
    auto words = stringTo1DArray<string>(input1);
    auto actual = ss.fullJustify(words, input2);
    bool passed = true;
    for (int i=0; i<(int)actual.size(); i++) {
        if ((int)actual[i].size() != input2) {
            passed = false;
            break;
        }
    }
    if (passed) {
        SPDLOG_INFO("Case({}, {}) passed", input1, input2);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: {}", input1, input2, numberVectorToString(actual));
    }
}


/*
Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.
Each letter in magazine can only be used once in ransomNote.

Example 1:
Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:
Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:
Input: ransomNote = "aa", magazine = "aab"
Output: true

Constraints:
1 <= ransomNote.length, magazine.length <= 105
ransomNote and magazine consist of lowercase English letters.
*/
bool Solution::canConstruct(string ransomNote, string magazine) {
    vector<int> count(128, 0);
    for (auto c: magazine) {
        count[c]++;
    }
    for (auto c: ransomNote) {
        count[c]--;
        if (count[c] < 0) {
            return false;
        }
    }
    return true;
}


void canConstruct_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    bool actual = ss.canConstruct(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given two strings s and t, determine if they are isomorphic.
Two strings s and t are isomorphic if the characters in s can be replaced to get t.
All occurrences of a character must be replaced with another character while preserving the order of characters.
No two characters may map to the same character, but a character may map to itself.

Example 1:
Input: s = "egg", t = "add"
Output: true

Example 2:
Input: s = "foo", t = "bar"
Output: false

Example 3:
Input: s = "paper", t = "title"
Output: true
 
Constraints:
1 <= s.length <= 5 * 104
t.length == s.length
s and t consist of any valid ascii character.
*/
bool Solution::isIsomorphic(string s, string t) {
    if (s.size() != t.size()) {
        return false;
    }
    int n = s.size();
    // since no two characters may map to the same character, we cannot use one map to record the mapping from s to t
    vector<int> ma(128, 0);
    vector<int> mb(128, 0);
    for (int i=0; i<n; i++) {
        if (ma[s[i]] != mb[t[i]]) {
            return false;
        }
        ma[s[i]] = i+1;
        mb[t[i]] = i+1;
    }
    return true;
}


void isIsomorphic_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    bool actual = ss.isIsomorphic(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running fullJustify tests: ");
    TIMER_START(fullJustify);
    fullJustify_scaffold("[This, is, an, example, of, text, justification.]", 16);
    fullJustify_scaffold("[What,must,be,acknowledgment,shall,be]", 16);
    fullJustify_scaffold("[Science,is,what,we,understand,well,enough,to,explain,to,a,computer.,Art,is,everything,else,we,do]", 20);
    fullJustify_scaffold("[a]", 1);
    TIMER_STOP(fullJustify);
    SPDLOG_WARN("fullJustify using {} ms", TIMER_MSEC(fullJustify));

    SPDLOG_WARN("Running isPalindrome tests: ");
    TIMER_START(isPalindrome);
    isPalindrome_scaffold("A man, a plan, a canal: Panama", 1);
    isPalindrome_scaffold("race a car", 0);
    isPalindrome_scaffold("", 1);
    isPalindrome_scaffold(" ", 1);
    isPalindrome_scaffold("a", 1);
    TIMER_STOP(isPalindrome);
    SPDLOG_WARN("isPalindrome using {} ms", TIMER_MSEC(isPalindrome));

    SPDLOG_WARN("Running canConstruct tests: ");
    TIMER_START(canConstruct);
    canConstruct_scaffold("a", "b", 0);
    canConstruct_scaffold("aa", "ab", 0);
    canConstruct_scaffold("aa", "aab", 1);
    canConstruct_scaffold("aa", "aba", 1);
    TIMER_STOP(canConstruct);
    SPDLOG_WARN("canConstruct using {} ms", TIMER_MSEC(canConstruct));

    SPDLOG_WARN("Running isIsomorphic tests: ");
    TIMER_START(isIsomorphic);
    isIsomorphic_scaffold("a", "a", 1);
    isIsomorphic_scaffold("a", "b", 1);
    isIsomorphic_scaffold("egg", "add", 1);
    isIsomorphic_scaffold("aa", "ab", 0);
    isIsomorphic_scaffold("foo", "bar", 0);
    isIsomorphic_scaffold("paper", "title", 1);
    isIsomorphic_scaffold("paper", "abadd", 0);
    TIMER_STOP(isIsomorphic);
    SPDLOG_WARN("isIsomorphic using {} ms", TIMER_MSEC(isIsomorphic));

}
