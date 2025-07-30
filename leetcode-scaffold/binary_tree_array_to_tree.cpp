#include "leetcode.h"

using namespace std;

class Solution {
public:
    TreeNode* constructFromInPost(vector<int>& inorder, vector<int>& postorder);
    TreeNode* constructFromPreIn(vector<int>& preorder, vector<int>& inorder);
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder);
};

/*
Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, construct and return the binary tree.

Example 1:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: inorder = [-1], postorder = [-1]
Output: [-1]
 
Constraints:
1 <= inorder.length <= 3000
postorder.length == inorder.length
-3000 <= inorder[i], postorder[i] <= 3000
inorder and postorder consist of unique values.
Each value of postorder also appears in inorder.
inorder is guaranteed to be the inorder traversal of the tree.
postorder is guaranteed to be the postorder traversal of the tree.
*/
TreeNode* Solution::constructFromInPost(vector<int>& inorder, vector<int>& postorder) {
    int sz = postorder.size();
    // Note that All the values of postorder and inorder are unique.
    map<int, int> inorder_pos_map; // element, element index in inorder
    for (int i=0; i<sz; i++) {
        inorder_pos_map[inorder[i]] = i;
    }
    // Note that in_r, post_r are inclusive
    function<TreeNode*(int, int, int, int)> dfs = [&] (int post_l, int post_r, int in_l, int in_r) {
        if (post_l > post_r) {
            return (TreeNode*)nullptr;
        }
        TreeNode* root = new TreeNode(postorder[post_r]);
        if (post_l == post_r) {
            return root;
        }
        int r = post_r;
        int in_i = inorder_pos_map[postorder[r]];
        // left subtree: [post_l, post_l+left_subtree_node_num-1]
        int left_subtree_node_num = in_i - in_l;
        root->left = dfs(post_l, post_l+left_subtree_node_num-1, in_l, in_i-1);
        // right subtree: [post_l+left_subtree_node_num, post_r-1]
        root->right = dfs(post_l+left_subtree_node_num, post_r-1, in_i+1, in_r); // `post_r-1` and `in_i+1` to exclude root node
        return root;
    };
    int post_l = 0;
    int post_r = sz-1; // inclusive
    int in_l = 0;
    int in_r = sz-1; // inclusive
    return dfs(post_l, post_r, in_l, in_r);
}


void constructFromInPost_scaffold(std::string input1, string input2, string expectedResult) {
    vector<int> in = stringTo1DArray<int>(input1);
    vector<int> post = stringTo1DArray<int>(input2);
    TreeNode* expected = stringToTreeNode(expectedResult);
    Solution ss;
    TreeNode* ans = ss.constructFromInPost(in, post);
    if (binaryTree_equal(ans, expected)) {
        SPDLOG_INFO("Case: ({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case: ({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        printBinaryTree(ans);
    }
}


/*
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]

Constraints:
1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.
*/
TreeNode* Solution::constructFromPreIn(vector<int>& preorder, vector<int>& inorder) {
    int sz = preorder.size();
    // Note that All the values of preorder and inorder are unique.
    map<int, int> inorder_pos_map; // element, element index in inorder
    for (int i=0; i<sz; i++) {
        inorder_pos_map[inorder[i]] = i;
    }
    // Note that pre_r, in_r are inclusive
    function<TreeNode*(int, int, int, int)> dfs = [&] (int pre_l, int pre_r, int in_l, int in_r) {
        if (pre_l > pre_r) {
            return (TreeNode*)nullptr;
        }
        TreeNode* root = new TreeNode(preorder[pre_l]);
        if (pre_l == pre_r) {
            return root;
        }
        int l = pre_l;
        int in_i = inorder_pos_map[preorder[l]];
        int left_subtree_node_num = in_i - in_l;
        // left subtree: [l+1, l+1+left_subtree_node_num-1]
        root->left = dfs(l+1, l+left_subtree_node_num, in_l, in_i-1);
        // right subtree: [l+left_subtree_node_num+1, pre_r]
        root->right = dfs(l+left_subtree_node_num+1, pre_r, in_i+1, in_r); // `in_i+1` to exclude root node
        return root;
    };
    int pre_l = 0;
    int pre_r = sz-1; // inclusive
    int in_l = 0;
    int in_r = sz-1; // inclusive
    return dfs(pre_l, pre_r, in_l, in_r);
}


void constructFromPreIn_scaffold(std::string input1, string input2, string expectedResult) {
    vector<int> pre = stringTo1DArray<int>(input1);
    vector<int> in = stringTo1DArray<int>(input2);
    TreeNode* expected = stringToTreeNode(expectedResult);
    Solution ss;
    TreeNode* ans = ss.constructFromPreIn(pre, in);
    if (binaryTree_equal(ans, expected)) {
        SPDLOG_INFO("Case: ({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case: ({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        printBinaryTree(ans);
    }
}


/*
Given two integer arrays, preorder and postorder where preorder is the preorder traversal of a binary tree of distinct values and postorder is the postorder traversal of the same tree, reconstruct and return the binary tree.
If there exist multiple answers, you can return any of them.

Example 1:
Input: preorder = [1,2,4,5,3,6,7], postorder = [4,5,2,6,7,3,1]
Output: [1,2,3,4,5,6,7]

Example 2:
Input: preorder = [1], postorder = [1]
Output: [1]

Constraints:
1 <= preorder.length <= 30
1 <= preorder[i] <= preorder.length
All the values of preorder are unique.
postorder.length == preorder.length
1 <= postorder[i] <= postorder.length
All the values of postorder are unique.
It is guaranteed that preorder and postorder are the preorder traversal and postorder traversal of the same binary tree.
*/
TreeNode* Solution::constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
    int sz = preorder.size();
    // Note that All the values of preorder and postorder are unique.
    map<int, int> postorder_pos_map; // element, element index in postorder
    for (int i=0; i<sz; i++) {
        postorder_pos_map[postorder[i]] = i;
    }
    // Note that pre_r, post_r are inclusive
    function<TreeNode*(int, int, int, int)> dfs = [&] (int pre_l, int pre_r, int post_l, int post_r) {
        if (pre_l > pre_r) {
            return (TreeNode*)nullptr;
        }
        TreeNode* root = new TreeNode(preorder[pre_l]);
        if (pre_l == pre_r) {
            return root;
        }
        int l = pre_l+1;
        int post_i = postorder_pos_map[preorder[l]];
        int left_subtree_node_num = post_i - post_l + 1;
        // left subtree: [l, l+left_subtree_node_num-1]
        root->left = dfs(l, l+left_subtree_node_num-1, post_l, post_i);
        // right subtree: [l+left_subtree_node_num, pre_r]
        root->right = dfs(l+left_subtree_node_num, pre_r, post_i+1, post_r-1); // `post_r-1` to exclude root node
        return root;
    };
    int pre_l = 0;
    int pre_r = sz-1; // inclusive
    int post_l = 0;
    int post_r = sz-1; // inclusive
    return dfs(pre_l, pre_r, post_l, post_r);
}


void constructFromPrePost_scaffold(std::string input1, string input2, string expectedResult) {
    vector<int> pre = stringTo1DArray<int>(input1);
    vector<int> post = stringTo1DArray<int>(input2);
    TreeNode* expected = stringToTreeNode(expectedResult);
    Solution ss;
    TreeNode* ans = ss.constructFromPrePost(pre, post);
    if (binaryTree_equal(ans, expected)) {
        SPDLOG_INFO("Case: ({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case: ({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        printBinaryTree(ans);
    }
}


int main() {
    SPDLOG_WARN("Running constructFromPrePost tests:");
    TIMER_START(constructFromPrePost);
    constructFromPrePost_scaffold("[1]", "[1]", "[1]");
    constructFromPrePost_scaffold("[1,2]", "[2,1]", "[1,2]");
    constructFromPrePost_scaffold("[1,2,4,5,3,6,7]", "[4,5,2,6,7,3,1]", "[1,2,3,4,5,6,7]");
    constructFromPrePost_scaffold("[1,2,4,5,3]", "[4,5,2,3,1]", "[1,2,3,4,5]");
    constructFromPrePost_scaffold("[1,2,4,5]", "[4,5,2,1]", "[1,2,null,4,5]");
    constructFromPrePost_scaffold("[3,4,1,2]", "[1,4,2,3]", "[3,4,2,1]");
    TIMER_STOP(constructFromPrePost);
    SPDLOG_WARN("constructFromPrePost using {} ms", TIMER_MSEC(constructFromPrePost));

    SPDLOG_WARN("Running constructFromPreIn tests:");
    TIMER_START(constructFromPreIn);
    constructFromPreIn_scaffold("[1]", "[1]", "[1]");
    constructFromPreIn_scaffold("[3,9,20,15,7]", "[9,3,15,20,7]", "[3,9,20,null,null,15,7]");
    constructFromPreIn_scaffold("[1,2,4,5,3,6,7]", "[4,2,5,1,6,3,7]", "[1,2,3,4,5,6,7]");
    constructFromPreIn_scaffold("[1,2,4,5,3]", "[4,2,5,1,3]", "[1,2,3,4,5]");
    constructFromPreIn_scaffold("[1,2,4,5]", "[4,2,5,1]", "[1,2,null,4,5]");
    constructFromPreIn_scaffold("[3,4,1,2]", "[1,4,3,2]", "[3,4,2,1]");
    TIMER_STOP(constructFromPreIn);
    SPDLOG_WARN("constructFromPreIn using {} ms", TIMER_MSEC(constructFromPreIn));

    SPDLOG_WARN("Running constructFromInPost tests:");
    TIMER_START(constructFromInPost);
    constructFromInPost_scaffold("[1]", "[1]", "[1]");
    constructFromInPost_scaffold("[9,3,15,20,7]", "[9,15,7,20,3]", "[3,9,20,null,null,15,7]");
    constructFromInPost_scaffold("[4,2,5,1,6,3,7]", "[4,5,2,6,7,3,1]", "[1,2,3,4,5,6,7]");
    constructFromInPost_scaffold("[4,2,5,1,3]", "[4,5,2,3,1]", "[1,2,3,4,5]");
    constructFromInPost_scaffold("[4,2,5,1]", "[4,5,2,1]", "[1,2,null,4,5]");
    constructFromInPost_scaffold("[1,4,3,2]", "[1,4,2,3]", "[3,4,2,1]");
    TIMER_STOP(constructFromInPost);
    SPDLOG_WARN("constructFromInPost using {} ms", TIMER_MSEC(constructFromInPost));
}
