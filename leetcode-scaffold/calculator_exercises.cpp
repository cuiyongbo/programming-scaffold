#include "leetcode.h"

using namespace std;

/* leetcode: 150, 224, 227, 772, 3, 223, 836, 189, 56 */
class Solution {
public:
    int calculate_224(string s);
    int calculate_227(string s);
    int calculate_772(string s);
    int evalRPN(vector<string>& tokens);

private:
    int calculate_227_infix2postfix(string s);
    int calculate_227_inplace(string s);

    // convert infix notation to postfix notation using Shunting-yard
    // refer to for further detail: https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    vector<string> infix_to_postfix_227(string s);
    int evaluate_postfix_notation(const vector<string>& tokens);

private:
    bool is_operator(char c) {
        return c == '+' || c == '-' ||
                        c == '*' || c == '/';
    }
    int evaluate(int op1, int op2, char op);

private:
    map<string, int> op_priority {
        {"+", 0}, {"-", 0},
        {"*", 1}, {"/", 1},
        {"a", 2}, // unary plus
        {"A", 2}, // unary minus
        {"(", INT32_MAX}, {")", INT32_MAX},
    };
};


int Solution::calculate_227_inplace(string s) {
    stack<int> operand_st;
    stack<string> operator_st;
    auto local_eval = [&] () {
        int op2 = operand_st.top(); operand_st.pop();
        int op1 = operand_st.top(); operand_st.pop();
        char op = operator_st.top()[0]; operator_st.pop();
        int res = evaluate(op1, op2, op);
        operand_st.push(res);
    };

    int sz = s.size();
    int left = -1, right = -1;
    for (int i=0; i<sz; ++i) {
        if (std::isdigit(s[i])) {
            if (left == -1) {
                left = right = i;
            } else {
                right = i;
            }
        } else if (is_operator(s[i])) {
            // save last operand
            operand_st.push(std::stoi(s.substr(left, right-left+1)));
            left = right = -1;

            string op = s.substr(i, 1);
            if (operator_st.empty()) {
                operator_st.push(op);
            } else if (op_priority[op] > op_priority[operator_st.top()]) {
                operator_st.push(op);
            } else {
                // 栈顶运算符的优先级不低于待插入运算符的优先级，
                // 需要循环计算当前栈顶的表达式，并保存运算结果
                // 直至新栈顶运算符优先级比待插入运算符的优先级低
                while (!operator_st.empty() && op_priority[op] <= op_priority[operator_st.top()]) {
                    local_eval();
                }
                operator_st.push(op);
            }
        } else {
            // space, go on
        }
    }
    if (left != -1) {
        // save last operand
        operand_st.push(std::stoi(s.substr(left, right-left+1)));
    }

    while (!operator_st.empty()) {
        local_eval();
    }
    return operand_st.top();
}


int Solution::calculate_227_infix2postfix(string s) {
    auto tokens = infix_to_postfix_227(s);
    return evaluate_postfix_notation(tokens);
}


int Solution::evaluate_postfix_notation(const vector<string>& tokens) {
    stack<int> st;
    for (const auto& t: tokens) {
        if (is_operator(t[0])) {
            // the order of stack is first-in, last-out
            int op2 = st.top(); st.pop();
            int op1 = st.top(); st.pop();
            st.push(evaluate(op1, op2, t[0]));
        } else if (t == "a") { // unary plus
            // do nothing
        } else if (t == "A") { // unary minus
            int op = st.top(); st.pop();
            st.push(-op);
        } else { // normal operand
            st.push(std::stoi(t));
        }
    }
    return st.top();
}


vector<string> Solution::infix_to_postfix_227(string s) {
    vector<string> rpn; // reverse polish notation
    stack<string> operator_st;
    int sz = s.size();
    int left = -1, right = -1;
    for (int i=0; i<sz; ++i) {
        if (std::isdigit(s[i])) {
            if (left == -1) {
                left = right = i;
            } else {
                right = i;
            }
        } else if (is_operator(s[i])) {
            // save last operand
            rpn.push_back(s.substr(left, right-left+1));
            left = right = -1;
            string op(1, s[i]); // current operator
            if (operator_st.empty()) {
                operator_st.push(op);
            } else if (op_priority[op] > op_priority[operator_st.top()]) { // current op has a higher priority than stack top's
                operator_st.push(op);
            } else {
                // current op has a lower priority than or the same as stack top's, we need evaluate the op(s) in the stack first, then current op
                while (!operator_st.empty() && op_priority[op] <= op_priority[operator_st.top()]) {
                    rpn.push_back(operator_st.top());
                    operator_st.pop();
                }
                operator_st.push(op);
            }
        } else {
            // whitespace, do nothing
        }
    }
    if (left != -1) { // save last operand
        rpn.push_back(s.substr(left, right-left+1));
    }
    while (!operator_st.empty()) { // save last operators
        rpn.push_back(operator_st.top());
        operator_st.pop();
    }
    SPDLOG_DEBUG("original equation: {}, RPN: {}", s, numberVectorToString(rpn));
    return rpn;
}


int Solution::evaluate(int op1, int op2, char op) {
    int res = 0;
    switch(op) {
        case '+':
            res = op1 + op2;
            break;
        case '-':
            res = op1 - op2;
            break;
        case '*':
            res = op1 * op2;
            break;
        case '/':
            res = op1 / op2;
            break;
        default:
            break;
    }
    return res;
}


/*
Implement a basic calculator to evaluate a simple expression string.
The expression string contains only non-negative integers, '+', '-', '*', '/' operators,
and open '(' and closing parentheses ')'. The integer division should truncate toward zero.
You may assume that the given expression is always valid. All intermediate results will be in the range of [-2^31, 2^31 - 1].
Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().
Constraints:
    1 <= s <= 10^4
    s consists of digits, '+', '-', '*', '/', '(', and ')'.
    s is a valid expression.
*/
int Solution::calculate_772(string s) {
    // 1. remove whitespaces in s
    string ns = s; s.clear();
    std::copy_if(ns.begin(), ns.end(), std::back_inserter(s), [](char c){return c != ' ';});
    // 2. convert infix notation to reverse polish notation
    vector<string> rpn;
    int left = -1;
    int right = -1;
    stack<string> op_st;
    int sz = s.size();
    for (int i=0; i<sz; i++) {
        if (std::isdigit(s[i])) {
            if (left == -1) {
                left = right = i;
            } else {
                right = i;
            }
        } else if (is_operator(s[i])) {
            if (left != -1) { // save last operand
                rpn.push_back(s.substr(left, right-left+1));
                left = right = -1;
            }
            string op(1, s[i]);
            if (op == "+") { // unary plus. such as +1, (+1), +(1+2)
                if (i==0 || s[i-1] == '(') {
                    op = "a";
                }
            }
            if (op == "-") { // unary minus.  such as -1, (-1), -(1+2)
                if (i==0 || s[i-1] == '(') {
                    op = "A";
                }
            }
            if (op_st.empty()) {
                op_st.push(op);
            } else if (op_priority[op] > op_priority[op_st.top()]) { // current op has a higher priority than stack top's
                op_st.push(op);
            } else {
                // current op has a lower priority than or the same as stack top's, then we need evaluate them first
                // because the element order of stack is last-in, first-out. the last operator will be evaluated first
                while (!op_st.empty() && op_priority[op] <= op_priority[op_st.top()]) {
                    if (op_st.top() == "(") { // we only remove '(' when meeting ')'
                        break;
                    }
                    rpn.push_back(op_st.top()); op_st.pop();
                }
                op_st.push(op);
            }
        } else if (s[i] == '(') {
            op_st.push("(");
        } else if (s[i] == ')') {
            // note that there is not '()' in rpn expression
            if (left != -1) { // save last operand
                rpn.push_back(s.substr(left, right-left+1));
                left = right = -1;
            }
            // evaluate ops inside '()'
            while (!op_st.empty() && op_st.top() != "(") {
                rpn.push_back(op_st.top()); op_st.pop();
            }
            op_st.pop(); // remove corresponding '('
        }
    }
    if (left != -1) { // save the final operand
        rpn.push_back(s.substr(left, right-left+1));
        left = right = -1;
    }
    // evaluate remaining ops
    while (!op_st.empty()) {
        rpn.push_back(op_st.top()); op_st.pop();
    }
    // 3. evaluate reverse polish notation
    return evaluate_postfix_notation(rpn);
}


/*
Given a string s which represents an expression, evaluate this expression and return its value.
The integer division should truncate toward zero.
You may assume that the given expression is always valid. All intermediate results will be in the range of [-(2^31), (2^31) - 1].
Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as ``eval()``.

Constraints:
    1 <= s.length <= 3 * 10^5
    s consists of integers and operators ('+', '-', '*', '/') separated by some number of spaces.
    '+' is not used as a unary operation (i.e., "+1" is invalid).
    '-' is not used as a unary operation (i.e., "-1" is invalid).
    s represents a valid expression.
    All the integers in the expression are non-negative integers in the range [0, 2^31 - 1].
    The answer is guaranteed to fit in a 32-bit integer.
Hint:
    solution 1. evaluate infix notation in-place
    solution 2. convert infix notation to postfix notation, then evaluate the converted notation
*/
int Solution::calculate_227(string s) {
    // return calculate_227_inplace(s);
    return calculate_227_infix2postfix(s);
}


/*
Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.
Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

Constraints:
    1 <= s.length <= 3 * 10^5
    s consists of digits, '+', '-', '(', ')', and ' '.
    s represents a valid expression.
    '+' is not used as a unary operation (i.e., "+1" and "+(2 + 3)" is invalid).
    '-' could be used as a unary operation (i.e., "-1" and "-(2 + 3)" is valid).
    There will be no two consecutive operators in the input.
    Every number and running calculation will fit in a signed 32-bit integer.
*/
int Solution::calculate_224(string s) {
    return calculate_772(s);
}


/*
Evaluate the value of an arithmetic expression in Reverse Polish Notation. (后缀表达式)
Valid operators are +, -, *, and /. Each operand may be an integer or another expression.
Note that division between two integers should truncate toward zero. (integer division)
It is guaranteed that the given RPN expression is always valid. That means the expression would
always evaluate to a result, and there will not be any division by zero operation.

Example 1:
    Input: tokens = ["2","1","+","3","*"]
    Output: 9
    Explanation: ((2 + 1) * 3) = 9
*/
int Solution::evalRPN(vector<string>& tokens) {
    auto is_operator = [] (string t) {
        string ops = "+-*/";
        return ops.find(t) != string::npos;
    };
    stack<int> st; // operands
    for (const auto& t: tokens) {
        if (is_operator(t)) {
            // they are all binary operators
            // note that the order of stack is first-in, last-out
            int t2 = st.top(); st.pop();
            int t1 = st.top(); st.pop();
            switch(t[0]) {
                case '+':
                    st.push(t1+t2);
                    break;
                case '-':
                    st.push(t1-t2);
                    break;
                case '*':
                    st.push(t1*t2);
                    break;
                case '/':
                    st.push(t1/t2); // integer division
                    break;
                default:
                    break;
            }
        } else { // operands
            st.push(std::stoi(t));
        }
    }
    return st.top();
}


void evalRPN_scaffold(string input, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<string>(input);
    int actual = ss.evalRPN(nums);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void calculate_scaffold(string input, int expectedResult, int func_no) {
    Solution ss;
    int actual = 0;
    if (func_no == 224) {
        actual = ss.calculate_224(input);
    } else if (func_no == 227) {
        actual = ss.calculate_227(input);
    } else if (func_no == 772) {
        actual = ss.calculate_772(input);
    } else {
        SPDLOG_ERROR("parameter error: test_func can only be [224, 227, 772]");
        return;
    }
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}, func_no={}) passed", input, expectedResult, func_no);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}, func_no={}) failed, actual={}", input, expectedResult, func_no, actual);
    }
}


int main() {
    SPDLOG_WARN("Running evalRPN tests:");
    TIMER_START(evalRPN);
    evalRPN_scaffold("[2,1,+,3,*]", 9);
    evalRPN_scaffold("[4,13,5,/,+]", 6);
    evalRPN_scaffold("[10,6,9,3,+,-11,*,/,*,17,+,5,+]", 22);
    TIMER_STOP(evalRPN);
    SPDLOG_WARN("evalRPN tests use {} ms", TIMER_MSEC(evalRPN));

    SPDLOG_WARN("Running calculate_227 tests:");
    TIMER_START(calculate_227);
    calculate_scaffold("1 + 1", 2, 227);
    calculate_scaffold(" 2-1 + 2", 3, 227);
    calculate_scaffold(" 6-4 / 2 ", 4, 227);
    calculate_scaffold("0", 0, 227);
    calculate_scaffold("3+2*2", 7, 227);
    calculate_scaffold(" 3/2 ", 1, 227);
    calculate_scaffold(" 3+5 / 2 ", 5, 227);
    calculate_scaffold("1*2-3/4+5*6-7*8+9/10", -24, 227);
    calculate_scaffold("1+2*3+4", 11, 227);
    TIMER_STOP(calculate_227);
    SPDLOG_WARN("calculate_227 tests use {} ms", TIMER_MSEC(calculate_227));

    SPDLOG_WARN("Running calculate_224 tests:");
    TIMER_START(calculate_224);
    calculate_scaffold("1 + 1", 2, 224);
    calculate_scaffold(" 2-1 + 2", 3, 224);
    calculate_scaffold("(1+(4+5+2)-3)+(6+8)", 23, 224);
    calculate_scaffold(" 6-4 / 2 ", 4, 224);
    calculate_scaffold("2*(5+5*2)/3+(6/2+8)", 21, 224);
    calculate_scaffold("(2+6* 3+5- (3*14/7+2)*5)+3", -12, 224);
    calculate_scaffold("0", 0, 224);
    calculate_scaffold("-1", -1, 224);
    calculate_scaffold("(-1)", -1, 224);
    calculate_scaffold("-(2 + 3)", -5, 224);
    TIMER_STOP(calculate_224);
    SPDLOG_WARN("calculate_224 tests use {} ms", TIMER_MSEC(calculate_224));

    SPDLOG_WARN("Running calculate_772 tests:");
    TIMER_START(calculate_772);
    calculate_scaffold("1 + 1", 2, 772);
    calculate_scaffold(" 2-1 + 2", 3, 772);
    calculate_scaffold("(1+(4+5+2)-3)+(6+8)", 23, 772);
    calculate_scaffold(" 6-4 / 2 ", 4, 772);
    calculate_scaffold("2*(5+5*2)/3+(6/2+8)", 21, 772);
    calculate_scaffold("(2+6* 3+5- (3*14/7+2)*5)+3", -12, 772);
    calculate_scaffold("0", 0, 772);
    calculate_scaffold("-1", -1, 772);
    calculate_scaffold("-(2 + 3)", -5, 772);
    calculate_scaffold("+1", 1, 772);
    calculate_scaffold("+(2 + 3)", 5, 772);
    TIMER_STOP(calculate_772);
    SPDLOG_WARN("calculate_772 tests use {} ms", TIMER_MSEC(calculate_772));
}
