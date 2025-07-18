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
    bool wordPattern(string pattern, string s);
    bool isAnagram(string s, string t);
    vector<vector<string>> groupAnagrams(vector<string>& strs);
    bool isHappy(int n);
    bool containsNearbyDuplicate(vector<int>& nums, int k);

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
                    if ((int)buffer.size() < maxWidth) {
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


/*
Given a pattern and a string s, find if s follows the same pattern.
Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s. Specifically:

Each letter in pattern maps to exactly one unique word in s.
Each unique word in s maps to exactly one letter in pattern.
No two letters map to the same word, and no two words map to the same letter.

Example 1:
Input: pattern = "abba", s = "dog cat cat dog"
Output: true
Explanation:
The bijection can be established as:
'a' maps to "dog".
'b' maps to "cat".

Example 2:
Input: pattern = "abba", s = "dog cat cat fish"
Output: false

Example 3:
Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false

Constraints:
1 <= pattern.length <= 300
pattern contains only lower-case English letters.
1 <= s.length <= 3000
s contains only lowercase English letters and spaces ' '.
s does not contain any leading or trailing spaces.
All the words in s are separated by a single space.
*/
bool Solution::wordPattern(string pattern, string s) {
    string item;
    std::stringstream ss(s);
    vector<string> words;
    while (std::getline(ss, item, ' ')) {
        words.push_back(item);
    }
    if (pattern.size() != words.size()) {
        return false;
    }
    map<char, string> d1;
    map<string, char> d2;
    int sz = pattern.size();
    for (int i=0; i<sz; i++) {
        if (d1.count(pattern[i]) && d1[pattern[i]] != words[i]) {
            return false;
        }
        if (d2.count(words[i]) && d2[words[i]] != pattern[i]) {
            return false;
        }
        d1[pattern[i]] = words[i];
        d2[words[i]] = pattern[i];
    }
    return true;
}


void wordPattern_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    bool actual = ss.wordPattern(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Constraints:
1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.

Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?
*/
bool Solution::isAnagram(string s, string t) {
    if (t.size() != s.size()) {
        return false;
    }
    vector<int> counting(128, 0);
    for (auto c: s) {
        counting[c]++;
    }
    for (auto c: t) {
        counting[c]--;
        if (counting[c]<0) {
            return false;
        }
    }
    return true;
}


void isAnagram_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    bool actual = ss.isAnagram(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]

Constraints:
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.
*/
vector<vector<string>> Solution::groupAnagrams(vector<string>& strs) {
if(0) { // naive solution
    int sz = strs.size();
    vector<bool> used(sz, false);
    vector<vector<string>> ans;
    for (int i=0; i<sz; i++) {
        if (used[i]) {
            continue;
        }
        vector<string> buffer;
        buffer.push_back(strs[i]);
        for (int j=i+1; j<sz; j++) {
            if (isAnagram(strs[i], strs[j])) {
                buffer.push_back(strs[j]);
                used[j] = true;
            }
        }
        used[i] = true;
        ans.push_back(buffer);
    }
    return ans;
}

{
    map<string, vector<string>> mp;
    for (auto w: strs) {
        auto k = w;
        std::sort(k.begin(), k.end());
        mp[k].push_back(w);
    }
    vector<vector<string>> ans;
    for (auto p: mp) {
        ans.push_back(p.second);
    }
    return ans;
}

}


void groupAnagrams_scaffold(string input1, string expectedResult) {
    Solution ss;
    auto words = stringTo1DArray<string>(input1);
    auto actual = ss.groupAnagrams(words);
    auto expected = stringTo2DArray<string>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input1, expectedResult);
        printf("actual.size: %d, expected.size: %d\n", (int)actual.size(), (int)expected.size());
        for (auto p: actual) {
            print_vector(p);
        }
    }
}


/*
Write an algorithm to determine if a number n is happy. A happy number is a number defined by the following process:

- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
- Those numbers for which this process ends in 1 are happy.

Return true if n is a happy number, and false if not.

Example 1:
Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

Example 2:
Input: n = 2
Output: false

Constraints:
1 <= n <= 2^31 - 1
*/
bool Solution::isHappy(int n) {
    set<int> nums;
    while (n!=1 && nums.count(n)!=1) {
        nums.insert(n);
        int tmp = 0;
        while (n > 0) {
            tmp += (n%10) * (n%10);
            n /= 10;
        }
        n = tmp;
    }
    return n==1;
}


void isHappy_scaffold(int input1, int expectedResult) {
    Solution ss;
    auto actual = ss.isHappy(input1);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input1, expectedResult, actual);
    }
}


/*
Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

Example 1:
Input: nums = [1,2,3,1], k = 3
Output: true

Example 2:
Input: nums = [1,0,1,1], k = 1
Output: true

Example 3:
Input: nums = [1,2,3,1,2,3], k = 2
Output: false
 
Constraints:
1 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9
0 <= k <= 105
*/
bool Solution::containsNearbyDuplicate(vector<int>& nums, int k) {
    map<int, int> mp; // val, index in nums
    for (int i=0; i<(int)nums.size(); i++) {
        if (mp.count(nums[i]) == 1) {
            if (i-mp[nums[i]] <= k) {
                return true;
            }
        }
        mp[nums[i]] = i;
    }
    return false;
}


void containsNearbyDuplicate_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input1);
    bool actual = ss.containsNearbyDuplicate(nums, input2);
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

    SPDLOG_WARN("Running wordPattern tests: ");
    TIMER_START(wordPattern);
    wordPattern_scaffold("abba", "dog cat cat dog", 1);
    wordPattern_scaffold("abba", "dog cat cat fish", 0);
    wordPattern_scaffold("aaaa", "dog cat cat dog", 0);
    TIMER_STOP(wordPattern);
    SPDLOG_WARN("wordPattern using {} ms", TIMER_MSEC(wordPattern));

    SPDLOG_WARN("Running isAnagram tests: ");
    TIMER_START(isAnagram);
    isAnagram_scaffold("abba", "dog cat cat dog", 0);
    isAnagram_scaffold("anagram", "nagaram", 1);
    TIMER_STOP(isAnagram);
    SPDLOG_WARN("isAnagram using {} ms", TIMER_MSEC(isAnagram));

    SPDLOG_WARN("Running groupAnagrams tests: ");
    TIMER_START(groupAnagrams);
    groupAnagrams_scaffold("[a]", "[[a]]");
    groupAnagrams_scaffold("[eat,tea,tan,ate,nat,bat]", "[[eat,tea,ate],[tan,nat],[bat]]");
    groupAnagrams_scaffold("[]", "[]");
    TIMER_STOP(groupAnagrams);
    SPDLOG_WARN("groupAnagrams using {} ms", TIMER_MSEC(groupAnagrams));

    SPDLOG_WARN("Running isHappy tests: ");
    TIMER_START(isHappy);
    isHappy_scaffold(1, 1);
    isHappy_scaffold(2, 0);
    isHappy_scaffold(19, 1);
    TIMER_STOP(isHappy);
    SPDLOG_WARN("isHappy using {} ms", TIMER_MSEC(isHappy));

    SPDLOG_WARN("Running containsNearbyDuplicate tests: ");
    TIMER_START(containsNearbyDuplicate);
    containsNearbyDuplicate_scaffold("[1,2,3,1]", 3, 1);
    containsNearbyDuplicate_scaffold("[1,0,1,1]", 1, 1);
    containsNearbyDuplicate_scaffold("[1,2,3,1,2,3]", 2, 0);
    containsNearbyDuplicate_scaffold("[1,2,3,1,2,3]", 3, 1);
    TIMER_STOP(containsNearbyDuplicate);
    SPDLOG_WARN("containsNearbyDuplicate using {} ms", TIMER_MSEC(containsNearbyDuplicate));

}
