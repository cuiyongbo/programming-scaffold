#include "leetcode.h"

using namespace std;

class Solution {
public:
    vector<long> getMinimumCosts(vector<int> warehouseCapacity, vector<vector<int>> additionHubs);
};

/*
warehouseCapacity is sorted in ascending order, warehouseCapacity[i] means the capcity of the i-th warehouse is warehouseCapacity[i]
the last warehouse is a hub to facilitate logistics.
the connection cost of a warehouse i is warehouseCapacity[j]-warehouseCapacity[i], where j>=i, and warehouse j is a hub. and the connection of a hub is 0.

we would add 2 addtional hubs at warehouse a, b (1-indexed) to facilitate logistics. each time
*/
vector<long> Solution::getMinimumCosts(vector<int> warehouseCapacity, vector<vector<int>> additionHubs) {

{
    vector<long> ans;
    long initial_cost = 0;
    int last_hub_cap = warehouseCapacity.back();
    int warehouse_num = warehouseCapacity.size();
    for (int i=0; i<warehouse_num; i++) {
        initial_cost += last_hub_cap-warehouseCapacity[i];
    }
    for (const auto& hubs: additionHubs) {
        long cost = initial_cost;
        cost -= (hubs[0]-0)*(last_hub_cap-warehouseCapacity[hubs[0]-1]);
        cost -= (hubs[1]-hubs[0])*(last_hub_cap-warehouseCapacity[hubs[1]-1]);
        ans.push_back(cost);
    }
    return ans;
}

}



int main() {

}

