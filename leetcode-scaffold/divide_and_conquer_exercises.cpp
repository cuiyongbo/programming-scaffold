#include "leetcode.h"

using namespace std;

/* leetcode: 75, 169, 654, 315 */
class Solution {
public:
    void sortColors(vector<int>& nums);
    int majorityElement(vector<int>& nums);
    TreeNode* constructMaximumBinaryTree(vector<int>& nums);
    vector<int> countSmaller(vector<int>& nums);
};


/*
Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, 
with the colors in the order red, white, and blue. We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
You must solve this problem without using the library's sort function.
Example 1:
    Input: nums = [2,0,2,1,1,0]
    Output: [0,0,1,1,2,2]
Example 2:
    Input: nums = [2,0,1]
    Output: [0,1,2]
*/
void Solution::sortColors(vector<int>& nums) {
    // hint: counting sort
    vector<int> counting(3, 0);
    for (auto n: nums) {
        counting[n]++;
    }
    nums.clear();
    for (int i=0; i<3; ++i) {
        nums.insert(nums.end(), counting[i], i);
    }
}

/*
Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
You may assume that the array is non-empty and the majority element always exist in the array.
*/
int Solution::majorityElement(vector<int>& nums) {
    // l and r are inclusive
    function<int(int,int)> dac = [&] (int l, int r) {
        if (l == r) { // trivial case
            return nums[l];
        }
        int m = (l+r)/2;
        int ml = dac(l, m);
        int mr = dac(m+1, r);
        if (ml == mr) {
            return ml;
        } else {
            // return the one with more occurrences
            return (std::count(nums.begin()+l, nums.begin()+r+1, ml) > 
                        std::count(nums.begin()+l, nums.begin()+r+1, mr)) ? ml : mr;
        }
    };
    return dac(0, nums.size()-1);
}

/*
Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:
    The root is the maximum number in the array.
    The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
    The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
    Construct the maximum tree by the given array and output the root node of this tree.
*/
TreeNode* Solution::constructMaximumBinaryTree(vector<int>& nums) {
    // r is not inclusive
    function<TreeNode*(int, int)> dac = [&] (int l, int r) {
        if (l > r) { // trivial case
            return (TreeNode*)nullptr;
        }
        int m = l;
        for (int i=l; i<=r; ++i) {
            if (nums[m] < nums[i]) {
                m = i;
            }
        }
        TreeNode* root = new TreeNode(nums[m]);
        root->left = dac(l, m-1);
        root->right = dac(m+1, r);
        return root;
    };
    return dac(0, nums.size()-1);
}


/*
You are given an integer array nums and you have to return a new counts array. 
The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].
*/
vector<int> Solution::countSmaller(vector<int>& nums) {
    struct bst_node {
        int val;
        int frequency; // how many times val occurs
        // NOTE: it is not fixed when creating the node, but just records the number of nodes residing left of node x during building the tree
        int left_node_count; // how many nodes reside in the left of val.
        bst_node* left;
        bst_node* right;
        bst_node(int v):
            val(v), frequency(1),
            left_node_count(0),
            left(nullptr), right(nullptr) {
        }
    };
    // return how many nodes resides the left of node val after inserting it
    auto bst_insert = [] (bst_node* root, int val) {
        int count = 0;
        bst_node* x = root;
        bst_node* px = nullptr; // parent of x
        bool found = false;
        while (x != nullptr) {
            px = x;
            if (x->val == val) {
                x->frequency++;
                count += x->left_node_count;
                found = true;
                break;
            } else if (x->val > val) { // val should go to the left of x
                x->left_node_count++;
                x = x->left;
            } else { // val shoud go to the right of x
                count += x->frequency;
                count += x->left_node_count;
                x = x->right;
            }
        }
        if (!found) {
            bst_node* t = new bst_node(val);
            //node->left_node_count = count; // NOTE: you cannot uncomment this line
            if (px->val > val) {
                px->left = t;
            } else {
                px->right = t;
            }
        }
        return count;
    };

    int sz = nums.size();
    vector<int> count(sz, 0);
    count[sz-1] = 0; // trivial case
    bst_node* root = new bst_node(nums[sz-1]); // trivial case
    for (int i=sz-2; i>=0; --i) {
        count[i] = bst_insert(root, nums[i]);
    }
    return count;
}


void majorityElement_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.majorityElement(nums);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, actual);
    }
}


void constructMaximumBinaryTree_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    TreeNode* root = ss.constructMaximumBinaryTree(vi);
    TreeNode* expected = stringToTreeNode(expectedResult);
    if(binaryTree_equal(root, expected)) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal:", input, expectedResult);
        printBinaryTree(root);
    }
}


void countSmaller_scaffold(string input, string expectedResult) {
    vector<int> vi = stringTo1DArray<int>(input);
    vector<int> expected = stringTo1DArray<int>(expectedResult);

    Solution ss;
    vector<int> actual = ss.countSmaller(vi);
    if(actual == expected) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, acutal={}", input, expectedResult, numberVectorToString(actual));
    }
}


void sortColors_scaffold(string input) {
    vector<int> vi = stringTo1DArray<int>(input);
    Solution ss;
    ss.sortColors(vi);
    if(std::is_sorted(vi.begin(), vi.end())) {
        SPDLOG_INFO("Case ({}) passed", input);
    } else {
        SPDLOG_ERROR("Case ({}) failed, acutal={}", input, numberVectorToString(vi));
    }
}

int main() {
    SPDLOG_WARN("Running majorityElement tests:");
    TIMER_START(majorityElement);
    majorityElement_scaffold("[6,1,2,8,6,4,5,3,6,6,6,5,6,6,6,6]", 6);
    majorityElement_scaffold("[6]", 6);
    majorityElement_scaffold("[6,1,6]", 6);
    // majorityElement_scaffold("[6,1]", -1);
    TIMER_STOP(majorityElement);
    SPDLOG_WARN("majorityElement tests use {} ms", TIMER_MSEC(majorityElement));

    SPDLOG_WARN("Running constructMaximumBinaryTree tests:");
    TIMER_START(constructMaximumBinaryTree);
    constructMaximumBinaryTree_scaffold("[1]", "[1]");
    constructMaximumBinaryTree_scaffold("[3,2,1,6,0,5]", "[6,3,5,null,2,0,null,null,1]");
    TIMER_STOP(constructMaximumBinaryTree);
    SPDLOG_WARN("constructMaximumBinaryTree tests use {} ms", TIMER_MSEC(constructMaximumBinaryTree));

    SPDLOG_WARN("Running countSmaller tests:");
    TIMER_START(countSmaller);
    countSmaller_scaffold("[5]", "[0]");
    countSmaller_scaffold("[5,5,5,5]", "[0,0,0,0]");
    countSmaller_scaffold("[5,2,6,1]", "[2,1,1,0]");
    countSmaller_scaffold("[5,1,1,-1,0]", "[4,2,2,0,0]");
    countSmaller_scaffold("[5,2,2,6,1]", "[3,1,1,1,0]");
    countSmaller_scaffold("[1,5,2,2,6,1]", "[0,3,1,1,1,0]");
    TIMER_STOP(countSmaller);
    SPDLOG_WARN("countSmaller tests use {} ms", TIMER_MSEC(countSmaller));

    SPDLOG_WARN("Running sortColors tests:");
    TIMER_START(sortColors);
    sortColors_scaffold("[2,1,1,0]");
    sortColors_scaffold("[2,0,2,1,1,0]");
    sortColors_scaffold("[2,0,1]");
    TIMER_STOP(sortColors);
    SPDLOG_WARN("sortColors tests use {} ms", TIMER_MSEC(sortColors));
}