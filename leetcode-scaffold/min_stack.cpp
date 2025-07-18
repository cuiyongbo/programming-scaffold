#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;

/* 
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.

Example 1:
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
Output
[null,null,null,null,-3,null,0,-2]
Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
 
Constraints:
-2^31 <= val <= 2^31 - 1
Methods pop, top and getMin operations will always be called on non-empty stacks.
At most 3 * 10^4 calls will be made to push, pop, top, and getMin.
*/

class MinStack {
private:
    stack<int> m_normal_stack;
    stack<int> m_min_stack;
public:
    MinStack() {
        m_min_stack.push(INT32_MAX);
    }

    void push(int v) {
        m_normal_stack.push(v);
        m_min_stack.push(min(v, m_min_stack.top()));
    }

    void pop() {
        m_normal_stack.pop();
        m_min_stack.pop();
    }

    int top() {
        return m_normal_stack.top();
    }

    int getMin() {
        return m_min_stack.top();
    }
};


void MinStack_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    MinStack tm;
    int n = (int)funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "push") {
            tm.push(stoi(funcArgs[i][0]));
            SPDLOG_INFO("{}({}) passed", funcOperations[i], funcArgs[i][0]);
        } else if (funcOperations[i] == "pop") {
            tm.pop();
            SPDLOG_INFO("{}() passed", funcOperations[i]);
        } else if (funcOperations[i] == "top" || funcOperations[i] == "getMin") {
            int actual = funcOperations[i] == "top" ? 
                                tm.top() :
                                tm.getMin();
            if (actual == stoi(ans[i])) {
                SPDLOG_INFO("{}() passed", funcOperations[i]);
            } else {
                SPDLOG_ERROR("{}() failed, expectedResult={}, actual={}", funcOperations[i], ans[i], actual);
            }
        }
    }
}


int main() {
    SPDLOG_WARN("Running MinStack tests:");
    TIMER_START(MinStack);
    MinStack_scaffold(
        "[MinStack,push,push,push,getMin,pop,top,getMin]",
        "[[],[-2],[0],[-3],[],[],[],[]]",
        "[null,null,null,null,-3,null,0,-2]");
    TIMER_STOP(MinStack);
    SPDLOG_WARN("MinStack tests use {} ms", TIMER_MSEC(MinStack));
}
