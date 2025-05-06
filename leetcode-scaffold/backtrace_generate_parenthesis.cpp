#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 20, 22, 301, 678 */
class Solution {
public:
    bool isValidParenthesisString_20(string s);
    bool isValidParenthesisString_678(string s);
    vector<string> generateParenthesis(int n);
    vector<string> removeInvalidParentheses(const string& s);
};


/*
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:
    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.
*/
bool Solution::isValidParenthesisString_20(string s) {
    map<char, char> m;
    m[')'] = '(';
    m[']'] = '[';
    m['}'] = '{';
    stack<char> st;
    for (auto c: s) {
        if (c == '(' || c == '{' || c == '[') { // push left bracket into the stack
            st.push(c);
        } else if (c == ')' || c == '}' || c == ']') { // test if there is a match for a right bracket
            if (st.empty() || st.top() != m[c]) {
                return false;
            }
            st.pop();
        }
    }
    return st.empty();
}


/*
Given a string containing only three types of characters: '(', ')' and '*', write a function to check
whether this string is valid. We define the validity of a string by these rules:
    Any left parenthesis '(' must have a corresponding right parenthesis ')'.
    Any right parenthesis ')' must have a corresponding left parenthesis '('.
    Left parenthesis '(' must go before the corresponding right parenthesis ')'.
    '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
An empty string is also valid.
*/
bool Solution::isValidParenthesisString_678(string s) {
    stack<int> left_st; // indices of left parentheses
    stack<int> wildcard_st; // indices of wildcards
    for (int i=0; i<s.size(); ++i) {
        if (s[i] == '(') {
            left_st.push(i);
        } else if (s[i] == '*') {
            wildcard_st.push(i);
        } else if (s[i] == ')') {
            if (!left_st.empty()) { // 1. test if there is a left parenthesis
                left_st.pop();
            } else if (!wildcard_st.empty()) { // 2. if not, test if there is a wildcard
                wildcard_st.pop();
            } else { // 3. if neither, then the string is not invalid
                return false;
            }
        }
    }
    // 4. for remaining '(' characters, we use wildcards as ')' characters to match them.
    // Note that a wildcard should come before a '(' character to match it
    while (!left_st.empty() && !wildcard_st.empty()) {
        if (left_st.top() < wildcard_st.top()) {
            left_st.pop(); wildcard_st.pop();
        } else {
            return false;
        }
    }
    // 5. take remaining wildcars as empty strings if there are any
    return left_st.empty();
}


/*
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses. For example, given n=3, a solution set is:
    [
        "((()))",
        "(()())",
        "(())()",
        "()(())",
        "()()()"
    ]
*/
vector<string> Solution::generateParenthesis(int n) {
    string alphabet = "()";
    string candidates;
    vector<string> ans;
    // diff = num_of_left_parentheses - num_of_right_parentheses
    function<void(int, int)> backtrace = [&] (int u, int diff) {
        if (u == 2*n) { // 1. termination
            if (diff == 0) {
                ans.push_back(candidates);
            }
            return;
        }
        for (auto c: alphabet) {
            int p = c=='(' ? 1 : -1;
            // 2. prune invalid branches
            if (diff+p < 0) { // there are more ')' than '('
                continue;
            }
            if (diff+p > n) { // the number of '(' is n at most
                continue;
            }
            // 3. perform backtrace
            candidates.push_back(c);
            backtrace(u+1, diff+p);
            candidates.pop_back();
        }
    };
    backtrace(0, 0);
    return ans;
}


/*
Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
Note: The input string may contain letters other than the parentheses ( and ).
*/
vector<string> Solution::removeInvalidParentheses(const string& s) {
if (0) {
    stack<int> left_st;
    stack<int> right_st;
    for (int i=0; i<(int)s.size(); i++) {
        if (s[i]=='(') {
            left_st.push(i);
        } else if (s[i]==')') {
            if (!left_st.empty()) {
                left_st.pop();
            } else {
                right_st.push(i);
            }
        }
    }
    // we have to remove characters in left_st and right_st
    // however we cannot find all answer in this way
    set<int> pos;
    while (!left_st.empty()) {
        pos.insert(left_st.top()); left_st.pop();
    }
    while (!right_st.empty()) {
        pos.insert(right_st.top()); right_st.pop();
    }
    string candidate;
    for (int i=0; i<(int)s.size(); i++) {
        if (pos.count(i)) {
            continue;
        }
        candidate.push_back(s[i]);
    }
    vector<string> ans;
    ans.push_back(candidate);
    return ans;
}

{
    int max_len = 0;
    string candidate;
    set<string> ans;
    // cur_index, numberOfLeftParenthesis, numberOfRightParenthesis
    function<void(int, int, int)> backtrace = [&] (int u, int l , int r) {
        // r > l means candidate is invalid, no need to go further
        if (r > l || u == s.length()) {
            // find a valid target, then save it when its size is no less than max_len
            if (r == l && candidate.size() >= max_len) { // termination
                if (candidate.size() > max_len) {
                    max_len = candidate.size();
                    ans.clear();
                }
                ans.insert(candidate);
            }
            return;
        }

        // option 1: discard s[u] if it is either '(' or ')'
        if (s[u] == '(' || s[u] == ')') {
            backtrace(u+1, l, r);
        }

        // option 2: keep s[u], normal backtrace
        // forward
        l += (s[u] == '(');
        r += (s[u] == ')');
        candidate.push_back(s[u]);
        backtrace(u+1, l, r);
        // restore
        candidate.pop_back();
        // not necessary, only for symmetry
        l -= s[u] == '(';
        r -= s[u] == ')';
    };
    backtrace(0, 0, 0);
    return vector<string>(ans.begin(), ans.end());
}

}


void isValidParenthesisString_scaffold(string input, bool expectedResult, int func) {
    Solution ss;
    bool actual = false;
    if (func == 20) {
        actual = ss.isValidParenthesisString_20(input);
    } else if (func == 678) {
        actual = ss.isValidParenthesisString_678(input);
    } else {
        SPDLOG_ERROR("parament error, func can ony be a value in [20, 678], actual={}", func);
        return;
    }
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}, func={}) passed", input, expectedResult, func);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}, func={}) failed, actual={}", input, expectedResult, func, actual);
    }
}


void generateParenthesis_scaffold(int input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.generateParenthesis(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    // to ease test
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for(const auto& s: actual) {
            cout << s << endl;
        }
    }
}


void removeInvalidParentheses_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.removeInvalidParentheses(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for(const auto& s: actual) {
            cout << s << endl;
        }
    }
}


int main() {
    SPDLOG_WARN("Running isValidParenthesisString tests: ");
    TIMER_START(isValidParenthesisString);
    isValidParenthesisString_scaffold("", true, 20);
    isValidParenthesisString_scaffold("(()", false, 20);
    isValidParenthesisString_scaffold("([])", true, 20);
    isValidParenthesisString_scaffold("(]", false, 20);
    isValidParenthesisString_scaffold("([)]", false, 20);
    isValidParenthesisString_scaffold("[((())), (()()), (())(), ()(()), ()()()]", true, 20);
    isValidParenthesisString_scaffold("", true, 678);
    isValidParenthesisString_scaffold("()", true, 678);
    isValidParenthesisString_scaffold("(*)", true, 678);
    isValidParenthesisString_scaffold("(*))", true, 678);
    isValidParenthesisString_scaffold("*(", false, 678);
    isValidParenthesisString_scaffold("*)", true, 678);
    isValidParenthesisString_scaffold("(*", true, 678);
    isValidParenthesisString_scaffold(")*", false, 678);
    TIMER_STOP(isValidParenthesisString);
    SPDLOG_WARN("isValidParenthesisString tests use {} ms", TIMER_MSEC(isValidParenthesisString));

    SPDLOG_WARN("Running generateParenthesis tests: ");
    TIMER_START(generateParenthesis);
    generateParenthesis_scaffold(1, "[()]");
    generateParenthesis_scaffold(2, "[(()), ()()]");
    generateParenthesis_scaffold(3, "[((())), (()()), (())(), ()(()), ()()()]");
    TIMER_STOP(generateParenthesis);
    SPDLOG_WARN("generateParenthesis tests use {} ms", TIMER_MSEC(generateParenthesis));

    SPDLOG_WARN("Running removeInvalidParentheses tests: ");
    TIMER_START(removeInvalidParentheses);
    removeInvalidParentheses_scaffold("()())()", "[(())(),()()()]");
    removeInvalidParentheses_scaffold("(a)())()", "[(a())(), (a)()()]");
    //removeInvalidParentheses_scaffold(")(", "[]");
    removeInvalidParentheses_scaffold(")()", "[()]");
    TIMER_STOP(removeInvalidParentheses);
    SPDLOG_WARN("removeInvalidParentheses tests use {} ms", TIMER_MSEC(removeInvalidParentheses));

}
