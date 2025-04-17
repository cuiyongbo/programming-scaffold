#include "leetcode.h"

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k);
};

int Solution::subarraySum(vector<int>& nums, int k) {
    int ans = 0;
    int sum = 0;
    std::unordered_map<int, int> sumMap;
    sumMap[0] = 1;
    for(auto n: nums) {
        sum += n;
        if(sumMap.find(sum-k) != sumMap.end()) {
            ans += sumMap[sum-k];
        }
        sumMap[sum]++;
    }
    return ans;
}
