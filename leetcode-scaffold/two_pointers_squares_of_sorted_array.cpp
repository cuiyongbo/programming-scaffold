#include "leetcode.h"

using namespace std;

/* leetcode: 977, 992, 172, 793*/
class Solution {
public:
    vector<int> sortedSquares(vector<int>& A);
    int subarraysWithKDistinct(vector<int>& A, int K);
    int trailingZeroes(int n);
    int preimageSizeFZF(int K);
};

/*
Given an array of integers A sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.
for example:
    Input: nums = [-4,-1,0,3,10]
    Output: [0,1,9,16,100]
    Explanation: After squaring, the array becomes [16,1,0,9,100].
    After sorting, it becomes [0,1,9,16,100].
*/
vector<int> Solution::sortedSquares(vector<int>& A) {
    // 1. split A into two partitions: A[:i]<0, A[i+1:]>=0
    // perform lower_bound to find the first non-negative element
    int sz = A.size();
    int l = 0;
    int r = sz;
    while (l < r) {
        int m = (l+r)/2;
        if (A[m] < 0) {
            l = m+1;
        } else {
            r = m;
        }
    }
    //SPDLOG_WARN("l={}, A[l]={}", l, A[l]);
    // 2. merge two partitions by squared values
    int i = l-1;
    int j = l;
    vector<int> ans(sz);
    for (int k=0; k<sz; k++) {
        if ((j==sz) || (i>=0 && A[i]*A[i]<A[j]*A[j])) {
            // i goes left
            ans[k] = A[i]*A[i];
            i--;
        } else {
            // j goes right
            ans[k] = A[j]*A[j];
            j++;
        }
    }
    return ans;
}


/*
    Given an integer array nums and an integer k, return the number of good subarrays of nums.
    A good array is an array where the number of different integers in that array is exactly k.
    For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3. A subarray is a contiguous part of an array.
    Constraints:
        1 <= nums[i], k <= nums.length
    Relative exercises:
        Longest Substring with At Most Two Distinct Characters
        Longest Substring with At Most K Distinct Characters
        Count Vowel Substrings of a String
        Number of Unique Flavors After Sharing K Candies
*/
int Solution::subarraysWithKDistinct(vector<int>& nums, int K) {
    // worker(k) means the number of subarrays with k or less than k distinct integer(s),
    // so the answer is worker(k)-worker(k-1)
    auto worker = [&] (int k) {
        int ans = 0;
        int sz = nums.size();
        // NOTE: we make sure that **1 <= nums[i], k <= nums.length**
        // count[i] means the count of i in `nums`
        vector<int> count(sz+1, 0);
        int distinct_nums = 0;
        for (int i=0,j=0; i<sz; ++i) {
            if (count[nums[i]] == 0) { // find a new distinct integer
                distinct_nums++;
            }
            count[nums[i]]++;
            assert(i>=j);
            // if there are more than `k` distinct numbers in nums[0:i]
            // shrink nums[j:i] so there are k or less than k distinct integer(s)
            for (; distinct_nums>k; j++) {
                --count[nums[j]];
                if (count[nums[j]] == 0) { // remove a distinct integer
                    distinct_nums--;
                }
            }
            ans += (i-j+1);
        }
        return ans;
    };
    return worker(K) - worker(K-1);
}


/*
Given an integer n, return the number of trailing zeroes in `n!`. Note that `n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1`.
for example:
Example 1:
Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.
Example 2:
Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.
Example 3:
Input: n = 0
Output: 0
Explanation: 0! = 1, no trailing zero.
*/
int Solution::trailingZeroes(int n) {
/*
the number of trailing zeros in the factorial of an integer n is equal to the number of times 10 is a factor in the product sequence.
and a factor of 10 is proudced by a pair of factors 2 and 5, since there are more factors of 2 than 5 in a product sequence. so the number of trailing zeros
is determined by the number of times 5.
*/
    int ans = 0;
    for (; n>0; n/=5) {
        ans += n/5;
    }
    return ans;
}


int Solution::preimageSizeFZF(int K) {
/*
    Let f(x) be the number of zeroes at the end of x!. (Recall that x! = 1 * 2 * 3 * ... * x, and by convention, 0! = 1.)
    For example, f(3) = 0 because 3! = 6 has no zeroes at the end, while f(11) = 2 because 11! = 39916800 has 2 zeroes at the end. 
    Given K, find how many non-negative integers x have the property that f(x) = K.
    Example 1:
        Input: K = 0
        Output: 5
        Explanation: 0!, 1!, 2!, 3!, and 4! end with K = 0 zeroes.
    Example 2:
        Input: K = 5
        Output: 0
        Explanation: There is no x such that x! ends in K = 5 zeroes.
    Hint: https://www.cnblogs.com/grandyang/p/9214055.html
*/
    // Oops!, the answer is either 0 or 5.
    auto numOfTrailingZeros = [&] (long m) {
        long res = 0;
        for (; m>0; m/=5) {
            res += m/5;
        }
        return res;
    };
    long l = 0;
    long r = 5L * (K+1);
    while (l < r) {
        long m = (l+r)/2;
        // if candidates exist, there must be 5 of them: i, i+1, i+2, i+3, i+4
        long cnt = numOfTrailingZeros(m);
        if (cnt == K) {
            return 5;
        } else if (cnt < K) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return 0;
}

void sortedSquares_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> nums = stringTo1DArray<int>(input);
    vector<int> actual = ss.sortedSquares(nums);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, numberVectorToString(actual));
    }
}

void subarraysWithKDistinct_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    int actual = ss.subarraysWithKDistinct(nums, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2,  expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}

void trailingZeroes_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.trailingZeroes(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void preimageSizeFZF_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.preimageSizeFZF(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running sortedSquares tests:");
    TIMER_START(sortedSquares);
    sortedSquares_scaffold("[1,2,3,4,5]", "[1,4,9,16,25]");
    sortedSquares_scaffold("[-5,-4,-3,-2,-1]", "[1,4,9,16,25]");
    sortedSquares_scaffold("[0,1,2,3,4,5]", "[0,1,4,9,16,25]");
    sortedSquares_scaffold("[-5,-4,-3,-2,-1,0]", "[0,1,4,9,16,25]");
    sortedSquares_scaffold("[-4,-1,0,3,10]", "[0,1,9,16,100]");
    sortedSquares_scaffold("[-7,-3,2,3,11]", "[4,9,9,49,121]");
    TIMER_STOP(sortedSquares);
    SPDLOG_WARN("sortedSquares tests use {} ms", TIMER_MSEC(sortedSquares));

    SPDLOG_WARN("Running subarraysWithKDistinct tests:");
    TIMER_START(subarraysWithKDistinct);
    subarraysWithKDistinct_scaffold("[1,2,1,2,3]", 2, 7);
    subarraysWithKDistinct_scaffold("[1,2,1,3,4]", 3, 3);
    subarraysWithKDistinct_scaffold("[1,2,3,4]", 3, 2);
    TIMER_STOP(subarraysWithKDistinct);
    SPDLOG_WARN("subarraysWithKDistinct tests use {} ms", TIMER_MSEC(subarraysWithKDistinct));

    SPDLOG_WARN("Running trailingZeroes tests:");
    TIMER_START(trailingZeroes);
    trailingZeroes_scaffold(0, 0);
    trailingZeroes_scaffold(5, 1);
    trailingZeroes_scaffold(3, 0);
    trailingZeroes_scaffold(10, 2);
    trailingZeroes_scaffold(15, 3);
    trailingZeroes_scaffold(16, 3);
    trailingZeroes_scaffold(20, 4);
    trailingZeroes_scaffold(21, 4);
    trailingZeroes_scaffold(25, 6);
    TIMER_STOP(trailingZeroes);
    SPDLOG_WARN("trailingZeroes tests use {} ms", TIMER_MSEC(trailingZeroes));

    SPDLOG_WARN("Running preimageSizeFZF tests:");
    TIMER_START(preimageSizeFZF);
    preimageSizeFZF_scaffold(0, 5);
    preimageSizeFZF_scaffold(5, 0);
    preimageSizeFZF_scaffold(3, 5);
    preimageSizeFZF_scaffold(4, 5);
    preimageSizeFZF_scaffold(1000000000, 5);
    TIMER_STOP(preimageSizeFZF);
    SPDLOG_WARN("preimageSizeFZF tests use {} ms", TIMER_MSEC(preimageSizeFZF));
}
