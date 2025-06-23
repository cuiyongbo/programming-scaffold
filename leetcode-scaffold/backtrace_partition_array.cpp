#include "leetcode.h"

using namespace std;

/* leetcode: 698, 93, 131, 241, 282, 842 */

namespace tiny_scaffold {
    int add(int a, int b) { return a+b;}
    int sub(int a, int b) { return a-b;};
    int mul(int a, int b) { return a*b;};
}

class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k);
    vector<string> restoreIpAddresses(string s);
    vector<vector<string>> partition(string s);
    vector<int> diffWaysToCompute(string input);
    vector<string> addOperators(string num, int target);
    vector<int> splitIntoFibonacci(string s);
};


/*
Given an array of integers nums and a positive integer k, find whether it’s possible to divide this array into k non-empty subsets whose sums are all equal.
For example, given inputs: nums = [4, 3, 2, 3, 5, 2, 1], k = 4, It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.
*/
bool Solution::canPartitionKSubsets(vector<int>& nums, int k) {
    // 1. check whether sum(nums) can be divided by k
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    int target = sum / k;
    if (target * k != sum) {
        return false;
    }
    // 2. check if nums can be divided into k partitions whose sum is equal to target
    int len = nums.size();
    vector<bool> used(len, false);
    function<bool(int, int)> backtrace = [&] (int cur_sum, int partition_num) {
        if (partition_num==k
            /*&& std::all_of(used.begin(), used.end(), [](bool v){return v;})*/ // not necessary?
        ) { // termination
            return true;
        }
        for (int i=0; i<len; ++i) {
            if (used[i] || cur_sum+nums[i]>target) { // prune invalid branches
                continue;
            }
            // perform backtrace
            used[i] = true;
            cur_sum += nums[i];
            if (cur_sum == target) {
                if (backtrace(0, partition_num+1)) {
                    return true;
                }
            } else {
                if (backtrace(cur_sum, partition_num)) {
                    return true;
                }
            }
            // restore last condition
            cur_sum -= nums[i];
            used[i] = false;
        }
        return false;
    };
    return backtrace(0, 0);
}


/*
A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros except zero itself.
For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.

Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s.
You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.
Example:
    Input: "25525511135"
    Output: ["255.255.11.135", "255.255.111.35"]
*/
vector<string> Solution::restoreIpAddresses(string input) {
    auto is_valid = [] (string tmp) {
        if (tmp.size()>1 && tmp[0] == '0') { // leading zero is invalid except zero itself
            return false;
        }
        int n = stoi(tmp);
        return 0<=n && n<=255;
    };
    int sz = input.size();
    vector<string> ans;
    vector<string> candidate;
    function<void(int)> backtrace = [&] (int u) {
        if (u == sz || candidate.size()>=4) { // termination
            if (u==sz && candidate.size() == 4) {
                ans.push_back(
                        candidate[0] + "." + 
                        candidate[1] + "." + 
                        candidate[2] + "." + 
                        candidate[3]);
            }
            return;
        }
        // the length of each sub-component is 3 at most
        for (int i=u; i<min(u+4, sz); ++i) {
            string tmp = input.substr(u, i-u+1); // 0< len(tmp) <= 3
            if (!is_valid(tmp)) { // prune invalid branches
                continue;
            }
            candidate.push_back(tmp);
            backtrace(i+1);
            candidate.pop_back();
        }
    };
    backtrace(0);
    return ans;
}


/*
Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
For example, given an input "aab", one possible output would be [["aa","b"],["a","a","b"]]
*/
vector<vector<string>> Solution::partition(string s) {

// sugar function
auto is_palindrome = [] (string input) {
    int sz = input.size();
    for (int i=0; i<sz/2; i++) {
        if (input[i] != input[sz-1-i]) {
            return false;
        }
    }        
    return true;
};

{
    int sz = s.size();
    vector<vector<string>> ans;
    vector<string> candidate;
    function<void(int)> backtrace = [&] (int u) {
        if (u == sz) { // termination
            ans.push_back(candidate);
            return;
        }
        for (int i=u; i<sz; i++) {
            auto sub = s.substr(u, i-u+1);
            if (!is_palindrome(sub)) { // prune invalid branches
                continue;
            }
            candidate.push_back(sub);
            backtrace(i+1);
            candidate.pop_back();
        }
    };
    backtrace(0);
    return ans;
}
    
{ // backtrace with memoization
    using result_type = vector<vector<string>>;
    using element_type = pair<bool,result_type>;
    map<string, element_type> sub_solution_mp;
    sub_solution_mp[""] = make_pair(true, result_type());
    function<element_type(string)> worker = [&] (string input) {
        if (sub_solution_mp.count(input) == 1) { // memoization
            return sub_solution_mp[input];
        }
        result_type ans;
        bool valid = false;
        for (int i=0; i<(int)input.size(); ++i) {
            auto left = input.substr(0, i+1);
            if (!is_palindrome(left)) { // prune invalid branches
                continue;
            }
            auto right = input.substr(i+1);
            auto rr = worker(right);
            if (rr.first) {
                valid = true;
                if (rr.second.empty()) {
                    vector<string> candidate;
                    candidate.insert(candidate.end(), left);
                    ans.push_back(candidate);
                }
                for (auto& p: rr.second) {
                    vector<string> candidate;
                    candidate.insert(candidate.end(), left);
                    candidate.insert(candidate.end(), p.begin(), p.end());
                    ans.push_back(candidate);
                }
            }
        }
        //cout << input << "(" << valid << ", " << ans.size() << ")" << endl;
        sub_solution_mp[input] = make_pair(valid, ans);
        return sub_solution_mp[input];
    };
    return worker(s).second;
}
    
}
    

/*
Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators.
The valid operators are +, - and *.
Example 1:
    Input: "2-1-1".
    ((2-1)-1) = 0
    (2-(1-1)) = 2
    Output: [0, 2]
Example 2:
    Input: "2*3-4*5"
    (2*(3-(4*5))) = -34
    ((2*3)-(4*5)) = -14
    ((2*(3-4))*5) = -10
    (2*((3-4)*5)) = -10
    (((2*3)-4)*5) = 10
    Output: [-34, -14, -10, -10, 10]
*/
vector<int> Solution::diffWaysToCompute(string input) {
    std::map<char, std::function<int(int, int)>> func_map;
    func_map['+'] = tiny_scaffold::add;
    func_map['-'] = tiny_scaffold::sub;
    func_map['*'] = tiny_scaffold::mul;
    using result_t = vector<int>;
    map<string, result_t> sub_solutions; // backtrace with memoization
    // return the possible values of expression input
    function<result_t(string)> backtrace = [&] (string expression) {
        if (sub_solutions.count(expression) != 0) { // memoization
            return sub_solutions[expression];
        }
        result_t ans;
        bool operator_found = false;
        for (int i=0; i<(int)expression.size(); ++i) {
            if (std::isdigit(expression[i])) {
                continue;
            }
            // input[i] is an operator, then we divide input into two sub-expressions input[0:i], input[i+1:]
            // and compute the value of them respectively
            operator_found = true;
            // we compute the value of input in a similiar post-order traversal
            string left_expression = expression.substr(0, i); // left sub-expression
            result_t left_val = backtrace(left_expression);
            string right_expression = expression.substr(i+1); // right sub-expression
            result_t right_val = backtrace(right_expression);
            for (auto a: left_val) {
                for (auto b: right_val) {
                    ans.push_back(func_map[expression[i]](a, b));
                }
            }
        }
        if (!operator_found) { // trivial case, input is an operand, such as "123"
            ans.push_back(std::stod(expression));
        }
        sub_solutions[expression] = ans;
        return ans;
    };
    return backtrace(input);
}


/*
Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.
Examples:
    "123", 6 -> ["1+2+3", "1*2*3"] 
    "232", 8 -> ["2*3+2", "2+3*2"]
    "105", 5 -> ["1*0+5","10-5"]
    "00", 0 -> ["0+0", "0-0", "0*0"]
    "3456237490", 9191 -> []
*/
vector<string> Solution::addOperators(string num, int target) {
    using result_t = vector<string>;
    result_t operators {"+", "-", "*"};
    map<string, int> priority_map;
    priority_map["+"] = 0;
    priority_map["-"] = 0;
    priority_map["*"] = 1;
    using operator_t = std::function<int(int, int)>;
    map<string, operator_t> func_map;
    func_map["+"] = tiny_scaffold::add;
    func_map["-"] = tiny_scaffold::sub;
    func_map["*"] = tiny_scaffold::mul;

    auto is_operator = [] (const string& c) { 
        return c == "-" || c == "+" || c =="*";
    };

    auto inplace_eval = [&] (result_t& infix_exp) {
        stack<int> operand;
        stack<string> op_st;
        for (auto& p: infix_exp) {
            if (is_operator(p)) {
                while (!op_st.empty() && priority_map[op_st.top()] >= priority_map[p]) {
                    auto b = operand.top(); operand.pop();
                    auto a = operand.top(); operand.pop();
                    operand.push(func_map[op_st.top()](a, b));
                    op_st.pop();
                }
                op_st.push(p);
            } else {
                operand.push(stoi(p));
            }
        }
        while(!op_st.empty()) {
            auto b = operand.top(); operand.pop();
            auto a = operand.top(); operand.pop();
            operand.push(func_map[op_st.top()](a, b));
            op_st.pop();
        }
        return operand.top();
    };

    // http://csis.pace.edu/~wolf/CS122/infix-postfix.htm
    auto infix_to_postfix = [&](const result_t& infix_exp) {
        stack<string> op;
        result_t postfix_exp;
        for (const auto& c: infix_exp) {
            if (is_operator(c)) {
                // pop stack when: 
                //      1. the priority of current operator is less than op.top's, such as + is against *
                //      2. the priority of current operator is equal to op.top's, such as + is against -
                if (!op.empty() && priority_map[op.top()] >= priority_map[c]) {
                    postfix_exp.push_back(op.top()); op.pop();
                }
                op.push(c);
            } else { // operand
                postfix_exp.push_back(c);
            }
        }
        while (!op.empty()) {
            postfix_exp.push_back(op.top()); op.pop();
        }
        return postfix_exp;
    };

    auto evaluate = [&](const result_t& exp) {
        stack<int> operands;
        const auto& postfix = infix_to_postfix(exp);
        for (const auto& op: postfix) {
            if (is_operator(op)) {
                // Note that stack order is first-in, last-out
                auto b = operands.top(); operands.pop();
                auto a = operands.top(); operands.pop();
                auto c = func_map[op](a, b);
                operands.push(c);
            } else {
                operands.push(std::stol(op));
            }
        }
        return operands.top();
    };

    result_t ans;
    result_t exp; // in-fix expression
    int len = num.size();
    function<void(int)> backtrace = [&] (int u) {
        if (u == len) { // termination
            //if (evaluate(exp) == target) {
            if (inplace_eval(exp) == target) {
                string path;
                for (auto& s: exp) {
                    path += s;
                }
                ans.push_back(path);
            }
            return;
        }
        for (int i=u; i<len; ++i) {
            // prune invalid branches
            if (i>u && num[u]=='0') { // skip operands with leading zero(s), such as "05"
                continue;
            }
            string cur = num.substr(u, i-u+1);
            if (stol(cur) > INT32_MAX) { // signed integer overflow
                continue;
            }
            exp.push_back(cur);
            if (i+1 == len) { // cur is the last operand, then evaluate the expression
                backtrace(i+1);
            } else {
                for (auto& op: operators) { // test each operator
                    exp.push_back(op);
                    backtrace(i+1);
                    exp.pop_back();
                }
            }
            exp.pop_back();
        }
    };

    backtrace(0);
    return ans;
}


/*
Given a string S of digits, such as S = "123456579", we can split it into a Fibonacci-like sequence [123, 456, 579].
Formally, a Fibonacci-like sequence is a list F of non-negative integers such that:
    0 <= F[i] <= 2^31 - 1, (that is, each integer fits a 32-bit signed integer type);
    F.length >= 3;
    and F[i] + F[i+1] = F[i+2] for all 0 <= i < F.length - 2.

Also note that when splitting the string into pieces, each piece must not have extra leading zeroes, except if the piece is the number 0 itself. 
also the input string contains only digits. Return any Fibonacci-like sequence split from S, or return [] if it cannot be done.

for example, 
    Input: "11235813"
    Output: [1,1,2,3,5,8,13]
    Input: "112358130"
    Output: []
    Explanation: The task is impossible.
    Input: "0123"
    Output: []
    Explanation: Leading zeroes are not allowed, so "01", "2", "3" is not valid.
*/
vector<int> Solution::splitIntoFibonacci(string input) {
    vector<int> ans;
    int sz = input.size();
    function<bool(int)> backtrace = [&] (int u) {
        if (u == sz) { // termination
            return ans.size() >= 3;
        }
        for (int i=u; i<sz; ++i) {
            // prune invalid branches
            if (i>u && input[u]=='0') { // skip element(s) with leading zero(s)
                continue;
            }
            string tmp = input.substr(u, i-u+1);
            long n = std::stol(tmp);
            if (n > INT32_MAX) { // prevent signed integer overflow
                continue;
            }
            int d = ans.size();
            if (d>=2 && (ans[d-2]+ ans[d-1] != n)) { // prune invalid branches
                continue;
            }
            ans.push_back(n);
            if (backtrace(i+1)) {
                return true;
            }
            ans.pop_back();
        }
        return false;
    };
    backtrace(0);
    return ans;
}


void canPartitionKSubsets_scaffold(string input1, int input2, bool expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input1);
    bool actual = ss.canPartitionKSubsets(g, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


void restoreIpAddresses_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.restoreIpAddresses(input);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for (const auto& s: actual) {
            cout << s << endl;
        }
    }
}


void partition_scaffold(string input, string expectedResult) {
    Solution ss;
    auto expected = stringTo2DArray<string>(expectedResult);
    auto actual = ss.partition(input);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for (const auto& s1: actual) {
            for (const auto& s2: s1) {
                std::cout << s2 << ",";
            }
            std::cout << std::endl;
        }
    }
}


void diffWaysToCompute_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.diffWaysToCompute(input);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, numberVectorToString<int>(actual));
    }
}


void addOperators_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.addOperators(input1, input2);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        for (const auto& s: actual) {
            cout << s << endl;
        }
    }
}


void splitIntoFibonacci_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<int> actual = ss.splitIntoFibonacci(input);
    if (!actual.empty() == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, numberVectorToString<int>(actual));
    }
}


int main() {
    SPDLOG_WARN("Running canPartitionKSubsets tests:");
    TIMER_START(canPartitionKSubsets);
    canPartitionKSubsets_scaffold("[4,3,2,3,5,2,1]", 4, true);
    TIMER_STOP(canPartitionKSubsets);
    SPDLOG_WARN("canPartitionKSubsets tests use {} ms", TIMER_MSEC(canPartitionKSubsets));

    SPDLOG_WARN("Running restoreIpAddresses tests:");
    TIMER_START(restoreIpAddresses);
    restoreIpAddresses_scaffold("25525511135", "[255.255.11.135, 255.255.111.35]");
    restoreIpAddresses_scaffold("1921684464", "[192.168.44.64]");
    restoreIpAddresses_scaffold("1921681100", "[19.216.81.100, 192.168.1.100, 192.16.81.100, 192.168.110.0]");
    restoreIpAddresses_scaffold("0000", "[0.0.0.0]");
    restoreIpAddresses_scaffold("101023", "[1.0.10.23, 1.0.102.3, 10.1.0.23, 10.10.2.3, 101.0.2.3]");
    restoreIpAddresses_scaffold("0011255245", "[]");
    TIMER_STOP(restoreIpAddresses);
    SPDLOG_WARN("restoreIpAddresses tests use {} ms", TIMER_MSEC(restoreIpAddresses));

    SPDLOG_WARN("Running partition tests:");
    TIMER_START(partition);
    partition_scaffold("aab", "[[a,a,b],[aa,b]]");
    partition_scaffold("a", "[[a]]");
    partition_scaffold("ab", "[[a, b]]");
    TIMER_STOP(partition);
    SPDLOG_WARN("partition tests use {} ms", TIMER_MSEC(partition));

    SPDLOG_WARN("Running diffWaysToCompute tests:");
    TIMER_START(diffWaysToCompute);
    diffWaysToCompute_scaffold("2-1-1", "[2,0]");
    diffWaysToCompute_scaffold("2*3-4*5", "[-34,-10,-14,-10,10]");
    TIMER_STOP(diffWaysToCompute);
    SPDLOG_WARN("diffWaysToCompute tests use {} ms", TIMER_MSEC(diffWaysToCompute));

    SPDLOG_WARN("Running splitIntoFibonacci tests:");
    TIMER_START(splitIntoFibonacci);
    splitIntoFibonacci_scaffold("123456579", true);
    splitIntoFibonacci_scaffold("11235813", true);
    splitIntoFibonacci_scaffold("112358130", false);
    splitIntoFibonacci_scaffold("0123", false);
    splitIntoFibonacci_scaffold("1101111", true);
    TIMER_STOP(splitIntoFibonacci);
    SPDLOG_WARN("splitIntoFibonacci tests use {} ms", TIMER_MSEC(splitIntoFibonacci));

    SPDLOG_WARN("Running addOperators tests:");
    TIMER_START(addOperators);
    addOperators_scaffold("123", 6, "[1+2+3, 1*2*3]");
    addOperators_scaffold("232", 8, "[2+3*2, 2*3+2]");
    addOperators_scaffold("105", 5, "[1*0+5, 10-5]");
    addOperators_scaffold("00", 0, "[0+0, 0-0, 0*0]");
    addOperators_scaffold("3456237490", 9191, "[]");
    TIMER_STOP(addOperators);
    SPDLOG_WARN("addOperators tests use {} ms", TIMER_MSEC(addOperators));
}
