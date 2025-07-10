#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    int strStr(string haystack, string pattern);
    bool isSubsequence(string s, string t);
    int lengthOfLongestSubstring(string str);
    string minWindow(string s, string t);
    string longestPalindrome(string s);
    bool isPalindrome(int x);


private:
    int strStr_naive(string haystack, string pattern);
    int strStr_kmp(string haystack, string pattern);

};


/*
Given two strings pattern and haystack, return the index of the first occurrence of pattern in haystack, or -1 if pattern is not part of haystack.

Example 1:
Input: haystack = "sadbutsad", pattern = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.

Example 2:
Input: haystack = "leetcode", pattern = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
 
Constraints:
1 <= haystack.length, pattern.length <= 104
haystack and pattern consist of only lowercase English characters.

Hint:
- Traverse
- Rabin-Karp algorighm
- KMP algorighm
*/
int Solution::strStr(string haystack, string pattern) {
    if (haystack.size() < pattern.size() || pattern.empty()) { // trivial case
        return -1;
    }
    // return strStr_naive(haystack, pattern);
    return strStr_kmp(haystack, pattern);
}


int Solution::strStr_naive(string haystack, string pattern) {
    for (int i=0; i<(int)haystack.size(); i++) {
        bool matched = true;
        for (int j=0; j<(int)pattern.size(); j++) {
            if (haystack[i+j] != pattern[j]) {
                matched = false;
                break;
            }
        }
        if (matched) {
            return i;
        }
    }
    return -1;
}


int Solution::strStr_kmp(string haystack, string pattern) {
    // 1. build LPS(the Longest Prefix-Suffix array)
    /*
    - For each position `i` in the pattern, `LPS[i]` stores the length of the longest proper prefix of the substring `pattern[0:i]` which is also a suffix of this substring pattern[0:i].
    - Proper prefix means prefix not equal to the whole substring.
    - This array helps to decide how many characters can be skipped after a mismatch.
    */
    int pattern_len = pattern.size();
    auto build_lps = [&] () {
        vector<int> lps(pattern_len, 0);
        for (int i=1; i<pattern_len; i++) {
            int len = i+1;
            for (int k=1; k<len; k++) {
                auto prefix = pattern.substr(0, k);
                auto suffix = pattern.substr(len-k, k); // suffix
                if (prefix == suffix) {
                    lps[i] = k;
                }
            }
        }
        return lps;
    };
    auto refined_build_lps = [&] () {
        vector<int> lps(pattern_len, 0);
        int i = 1;
        int len = 0; // length of the previous lps
        while (i < pattern_len) {
            if (pattern[i] == pattern[len]) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) {
                    len = lps[len-1]; // try to find shorter prefix-suffix
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        return lps;
    };
    //vector<int> lps = build_lps();
    vector<int> lps = refined_build_lps();
    //SPDLOG_INFO("LPS={}", numberVectorToString(lps));
    // 2. perform KMP search
    int i = 0;
    int j = 0;
    while (i < (int)haystack.size()) {
        if (haystack[i] == pattern[j]) {
            i++; j++;
            if (j == pattern_len) {
                //SPDLOG_INFO("i={}, j={}, pattern_len={}", i, j, pattern_len);
                return i - pattern_len;
            }
        } else {
            if (j > 0) {
                j = lps[j-1];
            } else {
                i++;
            }
        }
    }
    return -1;
}


void strStr_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    int actual = ss.strStr(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult: {}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult: {}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not). 

Example 1:
Input: s = "abc", t = "ahbgdc"
Output: true

Example 2:
Input: s = "axc", t = "ahbgdc"
Output: false
 
Constraints:
0 <= s.length <= 100
0 <= t.length <= 104
s and t consist only of lowercase English letters.
 
Follow up: Suppose there are lots of incoming s, say s1, s2, ..., sk where k >= 109, and you want to check one by one to see if t has its subsequence. In this scenario, how would you change your code?
*/
bool Solution::isSubsequence(string s, string t) {
    int m = s.size();
    int n = t.size();
    int i = 0;
    int j = 0;
    while (i<m && j<n) {
        if (s[i] == t[j]) {
            i++; j++;
        } else {
            j++;
        }
    }
    return i==m;
}


void isSubsequence_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    bool actual = ss.isSubsequence(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given a string s, find the length of the longest substring(not subsequence) without repeating characters.

Example 1:
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.
*/
int Solution::lengthOfLongestSubstring(string str) {
    int ans = 0;
    map<char, int> m; // char, the latest position of char
    int left = 0; // left boundary of substring without duplicate characters
    int sz = str.size();
    for (int i=0; i<sz; ++i) {
        if (m.count(str[i]) != 0) { // duplicate found
            left = max(left, m[str[i]]+1); // update left boundary. Note that we must keep `max()` since str[i] may appear in previous position
            m[str[i]] = i; // update occurrence of str[i] to the latest position
            ans = max(ans, i-left+1);
        } else {
            m[str[i]] = i; // update occurrence of str[i] to the latest position
            ans = max(ans, i-left+1);
        }
    }
    return ans;
}


void lengthOfLongestSubstring_scaffold(string input, int expectedResult) {
    Solution ss;
    int actual = ss.lengthOfLongestSubstring(input);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


/*
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.

Constraints:
m == s.length
n == t.length
1 <= m, n <= 105
s and t consist of uppercase and lowercase English letters.

Follow up: Could you find an algorithm that runs in O(m + n) time?
*/
string Solution::minWindow(string s, string t) {
    vector<int> need(128, 0); // char, occurrences
    for (auto c: t) {
        need[c]++;
    }
    vector<int> window(128, 0); // char, occurrences
    int m = s.size();
    int n = t.size();
    int cnt = 0;
    int j = 0;
    int k = -1;
    int mi = INT32_MAX;
    for (int i=0; i<m; i++) {
        window[s[i]]++;
        if (need[s[i]] >= window[s[i]]) { // we count s[i] no more than its occurrence in t
            ++cnt;
        }
        while (cnt == n) {
            if (i-j+1 < mi) {
                mi = i-j+1;
                k = j;
            }
            // 这两段代码的执行顺序不好理解
            if (need[s[j]] >= window[s[j]]) { // we need more s[j] in the subtring
                --cnt;
            }
            window[s[j]]--; j++;// decrease s[j]'s occurrence after moving j to right
        }
    }
    return  k==-1 ? "" : s.substr(k, mi);
}


void minWindow_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    string actual = ss.minWindow(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


/*
Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

Constraints:
1 <= s.length <= 1000
s consist of only digits and English letters.
*/
string Solution::longestPalindrome(string s) {
    int n = s.size();
    vector<vector<bool>> dp(n, vector<bool>(n, true));
    int ans = 0;
    int left = -1;
    // dp[i][j] means s[i:j] is palindromic
    // dp[i][j] = dp[i+1][j-1] and s[i]==s[j]
    for (int i=n-2; i>=0; i--) {
        for (int j=i+1; j<n; j++) {
            if (s[i] == s[j]) {
                dp[i][j] = dp[i+1][j-1];
                if (dp[i][j]) {
                    if (ans < j-i+1) {
                        ans = j-i+1;
                        left = i;
                    }
                }
            }
        }
    }
    return s.substr(left, ans);
}


void longestPalindrome_scaffold(string input, string expectedResult) {
    Solution ss;
    string actual = ss.longestPalindrome(input);
    if(actual == expectedResult || actual.size() == expectedResult.size()) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


/*
Given an integer x, return true if x is a palindrome, and false otherwise.

Example 1:
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

Example 2:
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.

Example 3:
Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
 
Constraints:
-231 <= x <= 231 - 1

Follow up: Could you solve it without converting the integer to a string?

Hint: given x, build y from LSB to MSB of x, y should be equal to x if x is a palindrome
*/
bool Solution::isPalindrome(int x) {
    // trivial cases:
    if (x<0) {
        return false;
    }
    int n = x;
    int y = 0;
    while (n>0) {
        int d = n%10;
        y = y*10 + d;
        n /= 10;
    }
    return x==y;
}


void isPalindrome_scaffold(int input, int expectedResult) {
    Solution ss;
    bool actual = ss.isPalindrome(input);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running strStr tests:");
    TIMER_START(strStr);
    strStr_scaffold("mississippi", "issi", 1);
    strStr_scaffold("mississippi", "issip", 4);
    strStr_scaffold("sad", "sadbutsad", -1);
    strStr_scaffold("leetcode", "leeto", -1);
    strStr_scaffold("sadbutsad", "sadbutsad", 0);
    strStr_scaffold("sadbutsad", "sad", 0);
    strStr_scaffold("nicetomeetyou", "nice", 0);
    strStr_scaffold("nicetomeetyou", "to", 4);
    strStr_scaffold("nicetomeetyou", "meet", 6);
    strStr_scaffold("nicetomeetyou", "you", 10);
    strStr_scaffold("ABABDABACDABABCABAB", "ABABCABAB", 10);
    strStr_scaffold("KMP algorithm is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a pattern string within a text string", "KMP", 0);
    strStr_scaffold("KMP algorithm is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a pattern string within a text string", "algorithm", 4);
    strStr_scaffold("KMP algorithm is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a pattern string within a text string", "pattern", 45);
    strStr_scaffold("KMP algorithm is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a pattern string within a text string", "pattern", 45);
    strStr_scaffold("KMP algorithm is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a pattern string within a text string", "occurrences", 96);
    strStr_scaffold("KMP algorithm is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a pattern string within a text string", "occurences", -1);
    TIMER_STOP(strStr);
    SPDLOG_WARN("strStr tests use {} ms", TIMER_MSEC(strStr));

    SPDLOG_WARN("Running isSubsequence tests: ");
    TIMER_START(isSubsequence);
    isSubsequence_scaffold("abc", "ahbgdc", 1);
    isSubsequence_scaffold("axc", "ahbgdc", 0);
    TIMER_STOP(isSubsequence);
    SPDLOG_WARN("isSubsequence using {} ms", TIMER_MSEC(isSubsequence));

    SPDLOG_WARN("Running lengthOfLongestSubstring tests:");
    TIMER_START(lengthOfLongestSubstring);
    lengthOfLongestSubstring_scaffold("", 0);
    lengthOfLongestSubstring_scaffold("abba", 2);
    lengthOfLongestSubstring_scaffold("ababab", 2);
    lengthOfLongestSubstring_scaffold("bbbbb", 1);
    lengthOfLongestSubstring_scaffold("abcdef", 6);
    lengthOfLongestSubstring_scaffold("pwwkew", 3);
    lengthOfLongestSubstring_scaffold("dvdf", 3);
    lengthOfLongestSubstring_scaffold("aaabcdddd", 4);
    lengthOfLongestSubstring_scaffold("aaabcddadd", 4);
    TIMER_STOP(lengthOfLongestSubstring);
    SPDLOG_WARN("lengthOfLongestSubstring tests use {} ms", TIMER_MSEC(lengthOfLongestSubstring));

    SPDLOG_WARN("Running minWindow tests:");
    TIMER_START(minWindow);
    minWindow_scaffold("ADOBECODEBANC", "ABC", "BANC");
    minWindow_scaffold("a", "a", "a");
    minWindow_scaffold("a", "aa", "");
    TIMER_STOP(minWindow);
    SPDLOG_WARN("minWindow tests use {} ms", TIMER_MSEC(minWindow));

    SPDLOG_WARN("Running longestPalindrome tests:");
    TIMER_START(longestPalindrome);
    longestPalindrome_scaffold("abba", "abba");
    longestPalindrome_scaffold("babad", "bab");
    longestPalindrome_scaffold("cbbd", "bb");
    TIMER_STOP(longestPalindrome);
    SPDLOG_WARN("longestPalindrome tests use {} ms", TIMER_MSEC(longestPalindrome));

    SPDLOG_WARN("Running isPalindrome tests:");
    TIMER_START(isPalindrome);
    isPalindrome_scaffold(121, true);
    isPalindrome_scaffold(123, false);
    isPalindrome_scaffold(10, false);
    isPalindrome_scaffold(-121, false);
    TIMER_STOP(isPalindrome);
    SPDLOG_WARN("isPalindrome tests use {} ms", TIMER_MSEC(isPalindrome));
    
}


/*
Q: tell me the detail about KMP algorithms

Sure! The **Knuth-Morris-Pratt (KMP) algorithm** is a classic string searching (pattern matching) algorithm that efficiently finds occurrences of a "pattern" string within a "text" string. It improves over the naive approach by avoiding unnecessary re-examination of characters.

---

## Overview of KMP Algorithm

- **Goal:** Find all occurrences of a pattern `P` of length `m` in a text `T` of length `n`.
- **Time complexity:** O(n + m)
- **Key idea:** Use information about the pattern itself to avoid redundant comparisons.

---

## Why KMP is efficient?

When a mismatch occurs, the naive algorithm moves the pattern by one and restarts matching from scratch, which can lead to O(n*m) worst case.

KMP uses a **preprocessed table** (called the **Longest Prefix Suffix (LPS) array**) to know how far to shift the pattern without re-checking characters that have already been matched.

---

## Key components

### 1. Longest Prefix Suffix (LPS) Array

- For each position `i` in the pattern, `LPS[i]` stores the length of the longest proper prefix of the substring `P[0..i]` which is also a suffix of this substring.
- Proper prefix means prefix not equal to the whole substring.
- This array helps to decide how many characters can be skipped after a mismatch.

---

### 2. The Search Process

- Compare characters of `P` with `T`.
- On mismatch, use the LPS array to shift the pattern efficiently.
- On match, continue checking next characters.
- When full pattern matches, record the position.

---

## Step-by-step example

Pattern: `ABABCABAB`

Text: `ABABDABACDABABCABAB`

---

## Building the LPS Array

| Index (i) | Pattern[i] | LPS[i] Explanation                    |
|-----------|------------|-------------------------------------|
| 0         | A          | 0 (no proper prefix/suffix)          |
| 1         | B          | 0                                   |
| 2         | A          | 1 (prefix "A" = suffix "A")          |
| 3         | B          | 2 ("AB")                            |
| 4         | C          | 0 (no match)                        |
| 5         | A          | 1 ("A")                            |
| 6         | B          | 2 ("AB")                           |
| 7         | A          | 3 ("ABA")                          |
| 8         | B          | 4 ("ABAB")                         |

So, LPS = `[0, 0, 1, 2, 0, 1, 2, 3, 4]`.

---

## KMP Search Algorithm Pseudocode

```python
def compute_lps(pattern):
    lps = [0] * len(pattern)
    length = 0  # length of previous longest prefix suffix
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    lps = compute_lps(pattern)
    i = 0  # index for text
    j = 0  # index for pattern
    results = []
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                results.append(i - j)  # match found
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return results
```

---

## Summary

| Aspect                | Description                            |
|-----------------------|------------------------------------|
| Purpose               | Search pattern in text efficiently   |
| Time Complexity       | O(n + m)                            |
| Space Complexity      | O(m) for LPS array                   |
| Key Idea              | Use LPS array to avoid redundant checks |
| Main Steps            | Preprocess pattern (LPS), then search |

---

If you want, I can walk you through a detailed example of how the algorithm processes a specific text and pattern step by step! Would you like that?

Q: what does LPS mean? tell me more

Great! Let's dive deeper into **LPS**, which stands for **Longest Prefix Suffix** — a crucial concept in the KMP (Knuth-Morris-Pratt) algorithm.

---

## What is LPS (Longest Prefix Suffix)?

For a given string (usually the pattern in KMP), the **LPS array** stores, for each position `i`, the length of the **longest proper prefix** of the substring `pattern[0..i]` that is also a **suffix** of this substring.

---

### Breaking down the terms:

- **Prefix:** A substring that starts at the beginning of the string.
- **Suffix:** A substring that ends at the end of the string.
- **Proper prefix:** A prefix that is **not equal to the entire substring**.
- **Proper suffix:** Similarly, a suffix that is not the entire substring.

---

### Example

Consider the pattern: `"ABABCABAB"`

Let's look at the substring up to index `i = 7` (0-based):

- Substring: `"ABABCABA"`

We want to find the **longest proper prefix** of `"ABABCABA"` that is also a suffix of `"ABABCABA"`.

- Proper prefixes: `"A"`, `"AB"`, `"ABA"`, `"ABAB"`, `"ABABC"`, `"ABABCA"`, `"ABABCAB"`
- Proper suffixes: `"A"`, `"BA"`, `"ABA"`, `"CABA"`, `"BCABA"`, `"ABCABA"`, `"BABCABA"`

The longest proper prefix that is also a suffix is `"ABA"` (length 3).

So, `LPS[7] = 3`.

---

## Why is LPS useful in KMP?

When a mismatch happens at position `j` in the pattern during matching, instead of restarting from the beginning of the pattern, we can use the LPS to know the next position in the pattern to compare.

This avoids re-checking characters that we know will match based on the prefix-suffix property.

---

## How to compute the LPS array?

We use a two-pointer approach:

- `length`: length of the previous longest prefix suffix.
- `i`: current position in the pattern.

Algorithm:

1. Initialize `lps[0] = 0`, `length = 0`, `i = 1`.
2. While `i < len(pattern)`:
   - If `pattern[i] == pattern[length]`, increment `length`, set `lps[i] = length`, increment `i`.
   - Else:
     - If `length != 0`, set `length = lps[length - 1]` (try shorter prefix).
     - Else, set `lps[i] = 0`, increment `i`.

---

## Example: Computing LPS for "ABABCABAB"

| i | pattern[i] | length | lps[i] | Explanation                          |
|---|------------|--------|--------|------------------------------------|
| 0 | A          | 0      | 0      | lps[0] always 0                    |
| 1 | B          | 0      | 0      | No prefix-suffix match             |
| 2 | A          | 0 → 1  | 1      | pattern[2] == pattern[0]           |
| 3 | B          | 1 → 2  | 2      | pattern[3] == pattern[1]           |
| 4 | C          | 2      | 0      | Mismatch, reset length             |
| 5 | A          | 0 → 1  | 1      | pattern[5] == pattern[0]           |
| 6 | B          | 1 → 2  | 2      | pattern[6] == pattern[1]           |
| 7 | A          | 2 → 3  | 3      | pattern[7] == pattern[2]           |
| 8 | B          | 3 → 4  | 4      | pattern[8] == pattern[3]           |

---

## Summary

| Concept               | Description                              |
|-----------------------|------------------------------------------|
| LPS[i]                | Length of longest proper prefix-suffix for substring ending at i |
| Purpose               | Helps avoid redundant comparisons in KMP |
| Computation complexity| O(m) where m = length of pattern         |
| Key insight           | Reuse previous prefix-suffix info to build LPS efficiently |

---

If you'd like, I can provide a detailed walkthrough of how LPS is used during the search phase in KMP or a code example!

Q: yes, please provide me a detailed walkthrough of how LPS is used during the search phase in KMP

Absolutely! Let me walk you through a **detailed example** of how the **LPS array** is used during the **search phase** of the KMP algorithm.

---

# Recap: What is LPS?

- The **LPS (Longest Prefix Suffix)** array tells us the longest proper prefix of the pattern that is also a suffix for every prefix of the pattern.
- It helps us **know where to resume matching in the pattern after a mismatch**, without re-examining characters unnecessarily.

---

# Example Setup

- **Text (T):** `"ABABDABACDABABCABAB"`
- **Pattern (P):** `"ABABCABAB"`

We already computed the LPS array for the pattern `P`:

```
Index:  0 1 2 3 4 5 6 7 8
P:      A B A B C A B A B
LPS:    0 0 1 2 0 1 2 3 4
```

---

# Step-by-step Search Walkthrough

We use two pointers:

- `i` for text `T`
- `j` for pattern `P`

---

### Initial state:

- `i = 0`, `j = 0`

---

### Iteration details:

| i | j | T[i] | P[j] | Action                                                                 | Explanation                                                                                  |
|---|---|-------|-------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| 0 | 0 | A     | A     | Match → i=1, j=1                                                      | Characters match, move both pointers forward                                                |
| 1 | 1 | B     | B     | Match → i=2, j=2                                                      | Continue matching                                                                           |
| 2 | 2 | A     | A     | Match → i=3, j=3                                                      | Continue matching                                                                           |
| 3 | 3 | B     | B     | Match → i=4, j=4                                                      | Continue matching                                                                           |
| 4 | 4 | D     | C     | Mismatch → j = LPS[j-1] = LPS[3] = 2                                 | Mismatch at P[4], shift pattern using LPS: move j back to 2 (pattern prefix length 2)        |
| 4 | 2 | D     | A     | Mismatch → j = LPS[j-1] = LPS[1] = 0                                 | Mismatch again, move j back to 0 (start of pattern)                                         |
| 4 | 0 | D     | A     | Mismatch → i=5                                                        | No partial match, move i forward in text                                                    |
| 5 | 0 | A     | A     | Match → i=6, j=1                                                      | Match found, move both pointers                                                             |
| 6 | 1 | B     | B     | Match → i=7, j=2                                                      | Continue matching                                                                           |
| 7 | 2 | A     | A     | Match → i=8, j=3                                                      | Continue matching                                                                           |
| 8 | 3 | C     | B     | Mismatch → j = LPS[j-1] = LPS[2] = 1                                 | Mismatch at P[3], shift j back to 1                                                        |
| 8 | 1 | C     | B     | Mismatch → j = LPS[j-1] = LPS[0] = 0                                 | Mismatch again, j=0                                                                        |
| 8 | 0 | C     | A     | Mismatch → i=9                                                        | No partial match, move i forward                                                           |
| 9 | 0 | D     | A     | Mismatch → i=10                                                       | No partial match, move i forward                                                           |
| 10| 0 | A     | A     | Match → i=11, j=1                                                     | Match found                                                                               |
| 11| 1 | B     | B     | Match → i=12, j=2                                                    | Continue matching                                                                         |
| 12| 2 | A     | A     | Match → i=13, j=3                                                    | Continue matching                                                                         |
| 13| 3 | B     | B     | Match → i=14, j=4                                                    | Continue matching                                                                         |
| 14| 4 | C     | C     | Match → i=15, j=5                                                    | Continue matching                                                                         |
| 15| 5 | A     | A     | Match → i=16, j=6                                                    | Continue matching                                                                         |
| 16| 6 | B     | B     | Match → i=17, j=7                                                    | Continue matching                                                                         |
| 17| 7 | A     | A     | Match → i=18, j=8                                                    | Continue matching                                                                         |
| 18| 8 | B     | B     | Match → i=19, j=9                                                    | Full pattern matched at index `i - j = 19 - 9 = 10`                                       |

---

# Key Points from the Walkthrough

- When characters match, both pointers advance.
- On mismatch:
  - Use `LPS[j-1]` to set `j` to the length of the longest prefix suffix.
  - Do **not** move `i` backward; only move forward or keep it the same.
- This avoids re-checking characters in the text that we know will match.
- When `j` reaches the length of the pattern, a full match is found.

---

# Why this is efficient?

- The text pointer `i` moves strictly forward.
- The pattern pointer `j` may jump back using LPS, but never moves forward beyond `i`.
- This guarantees **O(n + m)** time complexity, where `n` is the length of the text and `m` is the length of the pattern.

---

# Summary Table

| Situation           | Action on mismatch                           |
|---------------------|---------------------------------------------|
| `pattern[j] != text[i]` and `j != 0` | Set `j = LPS[j - 1]` (shift pattern using LPS) |
| `pattern[j] != text[i]` and `j == 0` | Move `i` forward (no partial match)         |

---

If you'd like, I can provide a fully annotated Python code example implementing this search with detailed comments! Would you like that?

Q: why we set j = LPS[j - 1] instead of j= LPS[j] during mismatch?

Excellent question! Understanding **why we set `j = LPS[j - 1]` instead of `j = LPS[j]` during a mismatch** is crucial to grasp the inner workings of the KMP algorithm.

---

## Recap: What is `j` and what is `LPS`?

- `j` is the current index in the **pattern** where we are trying to match.
- `LPS` is the **Longest Prefix Suffix** array, where `LPS[i]` gives the length of the longest proper prefix of the pattern substring `pattern[0..i]` that is also a suffix of this substring.

---

## Why `j = LPS[j - 1]` and not `LPS[j]`?

### 1. **`j` points to the current character in the pattern being matched**

- When a mismatch occurs at `pattern[j]` and `text[i]`, it means the character at `pattern[j]` does **not** match `text[i]`.
- Since `pattern[j]` failed, we want to find the **next best position in the pattern to resume matching**.
- The LPS array stores information **up to index `j-1`**, i.e., the longest prefix-suffix for the substring ending **before** the current character.

### 2. **`LPS[j]` is not defined yet or irrelevant**

- `LPS[j]` corresponds to the longest prefix-suffix for the substring ending at `j`.
- But since `pattern[j]` caused the mismatch, we cannot rely on `LPS[j]` to shift safely.
- The correct fallback is to look at `LPS[j - 1]`, which tells us the longest proper prefix-suffix length for the substring **just before** the mismatched character.

### 3. **Example**

Suppose `j = 4` (we are trying to match the 5th character in the pattern) and a mismatch happens.

- `LPS[3]` (which is `LPS[j - 1]`) tells us how many characters we can safely skip because they are both prefix and suffix.
- We set `j` to `LPS[3]` and try matching from there, **not** `LPS[4]`.

---

## Intuition

- The LPS array is built **based on prefixes ending before the current character**.
- When a mismatch occurs at `j`, the **longest prefix-suffix information up to `j-1`** guides how far to jump back.
- Using `LPS[j]` would be looking ahead or including the mismatched character, which is invalid.

---

## Summary

| Step                    | Explanation                                |
|-------------------------|--------------------------------------------|
| Mismatch at `pattern[j]`| Can't use `LPS[j]` because mismatch at `j` |
| Use `LPS[j - 1]`        | Gives longest prefix-suffix for substring ending at `j-1` |
| Set `j = LPS[j - 1]`    | Resume matching at this position in pattern |

---

If you'd like, I can provide a concrete example with numbers to illustrate this further! Want me to?

Q: Why can we safely skip the first `LPS[j-1]` characters in the pattern after a mismatch at position `j`?

Great follow-up! The core of your question is:

> **Why can we safely skip the first `LPS[j-1]` characters in the pattern after a mismatch at position `j`?**

---

## Intuition: Why skipping `LPS[j-1]` characters is safe

When a mismatch happens at `pattern[j]` and `text[i]`, it means the current character in the pattern doesn't match the current character in the text. Instead of restarting the pattern matching from the beginning, KMP uses the **LPS array** to avoid redundant checks.

### What does `LPS[j-1]` represent?

- `LPS[j-1]` is the length of the **longest proper prefix of the substring `pattern[0..j-1]` that is also a suffix of this substring**.
- This means the first `LPS[j-1]` characters of the pattern are **identical** to the last `LPS[j-1]` characters of the substring `pattern[0..j-1]`.
- and we are sure that the last `LPS[j-1]` characters of the substring `pattern[0..j-1]` match the last `LPS[j-1]` characters of `text[0..i-1]`
- as a result, the first `LPS[j-1]` characters of the pattern must also match the last `LPS[j-1]` characters of `text[0..i-1]`
- so we can skip the first `LPS[j-1]` characters of the pattern, and resume matching at `pattern[LPS[j-1]]`

---

### Why is this useful?

Because the prefix and suffix match, when a mismatch occurs at `j`, **the substring `pattern[0..LPS[j-1]-1]` has already been matched with the text** at some previous position.

- Since the prefix equals the suffix, the characters in the prefix are **guaranteed to match** the corresponding characters in the text.
- Therefore, instead of restarting at `pattern[0]`, we can **shift the pattern so that this prefix aligns with the suffix** we've just matched in the text.
- This means we can safely **skip the first `LPS[j-1]` characters** and resume matching at `pattern[LPS[j-1]]`.

---

### Visual example

Suppose:

- Pattern: `ABABCABAB`
- At position `j = 5` (0-based), mismatch occurs.
- `LPS[4]` (for substring `pattern[0..4] = "ABABC"`) is `0` → no proper prefix-suffix.
- So, we shift pattern to start matching at `pattern[0]`.

Now suppose:

- At position `j = 8`, mismatch occurs.
- `LPS[7]` (for substring `pattern[0..7] = "ABABCABA"`) is `3`.
- This means the first 3 characters `"ABA"` are both a prefix and suffix of the substring.
- So, instead of restarting at `pattern[0]`, we shift the pattern so that `pattern[3]` aligns with the current text position.
- We have already matched `"ABA"` before, so we skip rechecking those 3 characters.

---

## Summary

| Concept                       | Explanation                                  |
|------------------------------|----------------------------------------------|
| `LPS[j-1]`                   | Length of longest prefix-suffix at `j-1`    |
| Prefix == Suffix              | The first `LPS[j-1]` chars match last `LPS[j-1]` chars |
| Safe to skip                 | Because those chars are guaranteed matched   |
| Pattern shift on mismatch    | Resume matching at `pattern[LPS[j-1]]`       |

---

## Why this guarantees correctness?

- The LPS array encodes **all the information about repeated substrings** in the pattern.
- Using it ensures **no potential matches are missed**.
- It avoids redundant comparisons by leveraging the pattern’s internal structure.

---

If you want, I can illustrate this with a concrete step-by-step example on text and pattern showing exactly how the shift happens! Would you like that?

Q: how to compute LPS array?

Great! Computing the **LPS (Longest Prefix Suffix)** array is a fundamental step in the KMP algorithm. I'll walk you through the algorithm to compute the LPS array step-by-step, along with a clear example and a Python implementation.

---

## What is the LPS array?

For a pattern string `P` of length `m`, the LPS array `lps` is an array of length `m` where:

- `lps[i]` = length of the longest proper prefix of `P[0..i]` which is also a suffix of `P[0..i]`.

---

## Step-by-step algorithm to compute LPS

1. Initialize:
   - `lps[0] = 0` (the longest prefix-suffix for the first character is always 0)
   - Two pointers:
     - `length = 0` (length of the previous longest prefix suffix)
     - `i = 1` (current index in the pattern)

2. While `i < m`:
   - **If `P[i] == P[length]`:**
     - Increment `length` by 1
     - Set `lps[i] = length`
     - Increment `i` by 1
   - **Else (mismatch):**
     - If `length != 0`:
       - Set `length = lps[length - 1]` (try to find shorter prefix-suffix)
     - Else:
       - Set `lps[i] = 0`
       - Increment `i` by 1

---

## Example: Compute LPS for pattern `"ABABCABAB"`

| i | P[i] | length | lps[i] | Explanation                              |
|---|-------|--------|--------|------------------------------------------|
| 0 | A     | -      | 0      | Always 0 for first char                   |
| 1 | B     | 0      | 0      | `P[1] != P[0]`, length=0, lps[1]=0       |
| 2 | A     | 0 → 1  | 1      | `P[2] == P[0]`, length=1, lps[2]=1       |
| 3 | B     | 1 → 2  | 2      | `P[3] == P[1]`, length=2, lps[3]=2       |
| 4 | C     | 2      | 0      | `P[4] != P[2]`, length=2 → lps[1]=0, mismatch again, lps[4]=0 |
| 5 | A     | 0 → 1  | 1      | `P[5] == P[0]`, length=1, lps[5]=1       |
| 6 | B     | 1 → 2  | 2      | `P[6] == P[1]`, length=2, lps[6]=2       |
| 7 | A     | 2 → 3  | 3      | `P[7] == P[2]`, length=3, lps[7]=3       |
| 8 | B     | 3 → 4  | 4      | `P[8] == P[3]`, length=4, lps[8]=4       |

Final LPS: `[0, 0, 1, 2, 0, 1, 2, 3, 4]`

---

## Python code to compute LPS array

```python
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0  # length of the previous longest prefix suffix
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

# Example usage:
pattern = "ABABCABAB"
print(compute_lps(pattern))  # Output: [0, 0, 1, 2, 0, 1, 2, 3, 4]
```

---

## Summary

- The LPS array helps KMP skip unnecessary comparisons.
- It is computed in **O(m)** time.
- Uses two pointers to track prefix-suffix matches efficiently.

---

If you want, I can also show how this LPS array is used during the actual pattern search step!
*/