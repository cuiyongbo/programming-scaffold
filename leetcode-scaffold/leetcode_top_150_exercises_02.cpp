#include "leetcode.h"

using namespace std;

// https://leetcode.com/studyplan/top-interview-150/
class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth);
    

};


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


int main() {
    SPDLOG_WARN("Running fullJustify tests: ");
    TIMER_START(fullJustify);
    fullJustify_scaffold("[This, is, an, example, of, text, justification.]", 16);
    fullJustify_scaffold("[What,must,be,acknowledgment,shall,be]", 16);
    fullJustify_scaffold("[Science,is,what,we,understand,well,enough,to,explain,to,a,computer.,Art,is,everything,else,we,do]", 20);
    fullJustify_scaffold("[a]", 1);
    TIMER_STOP(fullJustify);
    SPDLOG_WARN("fullJustify using {} ms", TIMER_MSEC(fullJustify));
}
