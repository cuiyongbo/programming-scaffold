#include "leetcode.h"

using namespace std;

/* leetcode: 323, 399, 721, 737, 839, 952, 924, 990 */
class Solution {
public:
    int countComponents(int n, vector<vector<int>>& edges);
    int largestComponentSize(vector<int>& A);
    int numSimilarGroups(vector<string>& A);
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries);
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts);
    bool areSentencesSimilar(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs);
    bool equationsPossible(vector<string>& equations);
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial);
};


/*
You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi (0-indexed) in the graph. (bidirectional graph)
Return the number of connected components in the graph.
*/
int Solution::countComponents(int n, vector<vector<int>>& edges) {

if (0) { // dfs solution
    vector<vector<int>> graph(n);
    for (auto& e: edges) {
        graph[e[0]].push_back(e[1]);
        graph[e[1]].push_back(e[0]);
    }
    vector<int> visited(n, 0);
    // mark one connected component starting with node u
    function<void(int)> dfs = [&] (int u) {
        visited[u] = 1; // visiting
        for (auto v: graph[u]) {
            if (visited[v] == 0) { // unvisited
                dfs(v);
            }
            // it doesn't matter whether there is a cycle or not
        }
        visited[u] = 2; // visited
    };
    int ans = 0;
    // iterate over all nodes
    for (int i=0; i<n; ++i) {
        if (visited[i] == 0) {
            dfs(i);
            ans++;
        }
    }
    return ans;
}

{ // disjoint set solution
    DisjointSet dsu(n);
    for (auto& e: edges) {
        dsu.unionFunc(e[0], e[1]);
    }
    set<int> groups;
    for (int i=0; i<n; i++) {
        groups.insert(dsu.find(i));
    }
    return groups.size();
}

}


/*
Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). 
Given some queries, return the answers. If the answer does not exist, return -1.0.
Example:
    Given a / b = 2.0, b / c = 3.0.
    queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
    return [6.0, 0.5, -1.0, 1.0, -1.0].
The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries,
where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.
*/
vector<double> Solution::calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
    // build a bidirectional graph
    using key_t =  std::pair<string, string>;
    map<key_t, double> weight_map;
    map<string, vector<string>> graph;
    for (int i=0; i<(int)equations.size(); ++i) {
        auto& p = equations[i];
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
        weight_map[{p[0], p[1]}] = values[i];
        weight_map[{p[1], p[0]}] = 1/values[i];
    }
    vector<key_t> path;
    set<string> visited;
    // return true if end is reachable from u
    function<bool(string, string)> dfs = [&] (string u, string end) {
        if (u == end) {
            return true;
        }
        visited.insert(u);
        for (auto v: graph[u]) {
            if (visited.count(v) == 0) {
                // perform backtrace
                path.emplace_back(u, v);
                if (dfs(v, end)) {
                    return true;
                }
                path.pop_back();
            }
        }
        return false;
    };
    auto eval_path = [&] (vector<key_t>& path) {
        double val = 1.0;
        for (auto& p: path) {
            val *= weight_map[p];
        }
        return val;
    };
    // use dfs to find the path from start to end
    function<double(string, string)> calc_query = [&] (string start, string end) {
        // return -1 if either start or end does not exist in the graph
        if (graph.count(start)==0 || graph.count(end)==0) { // trivial cases
            return -1.0;
        }
        // return 1.0 if start == end
        if (start == end) { // trivial cases
            return 1.0;
        }
        visited.clear();
        path.clear();
        if (dfs(start, end)) {
            return eval_path(path);
        } else {
            return -1.0;
        }
    };
    // traverse over all queries to calculate answers
    vector<double> ans;
    for (auto& p: queries) {
        ans.push_back(calc_query(p[0], p[1]));
    }
    return ans;
}


/*
Given two sentences words1, words2 (each represented as an array of strings), and a list of similar word pairs pairs, determine if two sentences are similar.

For example, words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar, 
if the similar word pairs are pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]].

Note that the similarity relation is transitive. For example, if “great” and “good” are similar, and “fine” and “good” are similar, then “great” and “fine” are similar.
Similarity is also symmetric. For example, “great” and “fine” being similar is the same as “fine” and “great” being similar. (bidirectonal graph)

Also, a word is always similar with itself. For example, the sentences words1 = ["great"], words2 = ["great"], 
pairs = [] are similar, even though there are no specified similar word pairs.

Two sentences are similar if:
    They have the same length (i.e., the same number of words)
    sentence1[i] and sentence2[i] are similar.
*/
bool Solution::areSentencesSimilar(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs) {
    if (words1.size() != words2.size()) { // two similar sentence must have the same length
        return false;
    }
    // build a bidirectional graph
    map<string, vector<string>> graph;
    for (auto& p: pairs) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }
    set<string> visited;
    // return true if end is reachable from start
    function<bool(string,string)> dfs = [&] (string start, string end) {
        if (start == end) {
            return true;
        }
        visited.insert(start);
        for (auto& v: graph[start]) {
            if (visited.count(v) == 0) {
                if (dfs(v, end)) {
                    return true;
                }
            }
        }
        return false;
    };
    for (int i=0; i<(int)words1.size(); ++i) {
        visited.clear();
        if (!dfs(words1[i], words2[i])) { // perform dfs for each pair
            return false;
        }
    }
    return true;
}


/*
Given a list accounts, each element accounts[i] is a list of strings, 
where the first element accounts[i][0] is a name, and the rest of 
elements are emails representing emails of the account. ([name: email_list])

Now, we would like to merge these accounts. **Two accounts definitely belong to the same person
if there is some email that is common to both accounts.** Note that even if two accounts have the same name, 
they may belong to different people as people could have the same name. 
A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: 
the first element of each account is the name, and the rest of the elements are emails in sorted order. 
The accounts themselves can be returned in any order.

Hint: 
    solution one: use dfs to find connected components
    solution two: use disjoint_set to find connected components
*/
vector<vector<string>> Solution::accountsMerge(vector<vector<string>>& accounts) {

{ // dfs solution
    int account_num = accounts.size();
    map<string, vector<int>> email_map; // email, account ids
    for (int i=0; i<account_num; i++) {
        int sz = accounts[i].size();
        for (int j=1; j<sz; j++) {
            email_map[accounts[i][j]].push_back(i);
        }
    }
    // build a bidirectional graph
    vector<vector<int>> graph(accounts.size());
    for (auto p: email_map) {
        int sz = p.second.size();
        if (sz < 2) {
            continue;
        }
        int u = p.second[0];
        for (int i=1; i<sz; i++) {
            int v = p.second[i];
            graph[u].push_back(v);
            graph[v].push_back(u);
            u = v;
        }
    }
    // find CCs
    vector<int> path;
    vector<int> visited(account_num, 0);
    function<void(int)> dfs = [&] (int u) {
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                dfs(v);
            }
        }
        visited[u] = 2;
        path.push_back(u);
    };
    // cluster groups
    map<int, vector<int>> groups;
    for (int u=0; u<account_num; u++) {
        if (visited[u] == 0) {
            dfs(u);
            groups[u] = path;
            path.clear();
        }
    }
    // merge groups
    vector<vector<string>> ans;
    for (auto p: groups) {
        string name = accounts[p.first][0];
        set<string> emails;
        for (auto n: p.second) {
            emails.insert(std::next(accounts[n].begin()), accounts[n].end());
        }
        vector<string> one_user;
        one_user.push_back(name);
        one_user.insert(one_user.end(), emails.begin(), emails.end());
        ans.push_back(one_user);
    }
    return ans;
}

{ // dis-joint set solution
    auto is_same_account = [&] (int i, int j) {
        for (int u=1; u<(int)accounts[i].size(); u++) {
            for (int v=1; v<(int)accounts[j].size(); v++) {
                if (accounts[i][u] == accounts[j][v]) {
                    return true;
                }
            }
        }
        return false;
    };

    // build dsu
    int n = accounts.size();
    DisjointSet dsu(n);
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            if (is_same_account(i, j)) {
                dsu.unionFunc(i, j);
            }
        }
    }
    // cluster each group
    map<int, vector<int>> groups;
    for (int i=0; i<n; i++) {
        int g = dsu.find(i);
        groups[g].push_back(i);
    }
    // merge accounts by group
    vector<vector<string>> ans;
    for (auto& p: groups) {
        set<string> emails;
        for (auto i: p.second) {
            emails.insert(std::next(accounts[i].begin()), accounts[i].end());
        }
        vector<string> buffer;
        buffer.push_back(accounts[p.first][0]); // name
        buffer.insert(buffer.end(), emails.begin(), emails.end()); // emails
        ans.push_back(buffer);
    }
    return ans;
}

}


/*
Two strings X and Y are similar if we can swap two letters (in different positions) of X, so that it equals Y.
For example, "tars" and "rats" are similar (swapping at positions 0 and 2), and "rats" and "arts" are similar, but "star" is not similar to "tars", "rats", or "arts".

Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}. Notice that "tars" and "arts" are in the same group even though they are not similar.  
Formally, each group is such that a word is in the group if and only if it is similar to at least one other word in the group.

We are given a list A of strings. Every string in A is an anagram of every other string in A. How many groups are there?

Hint: use dfs/disjoint_set to find connected components
*/
int Solution::numSimilarGroups(vector<string>& A) {
    auto is_similar = [&] (string u, string v) {
        int diff = 0;
        for (int i=0; i<(int)u.size(); ++i) {
            if (u[i] != v[i]) {
                diff++;
            }
        }
        return diff == 2;
    };

if (0) { // disjoint_set solution
    int n = A.size();
    DisjointSet dsu(n);
    for (int i=0; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            if (is_similar(A[i], A[j])) {
                dsu.unionFunc(i, j);
            }
        }
    }
    set<int> groups;
    for (int i=0; i<n; ++i) {
        groups.insert(dsu.find(i));
    }
    return groups.size();
}

{ // dfs solution
    // build a bidirectional graph
    int sz = A.size();
    map<string, vector<string>> graph;
    for (int i=0; i<sz; ++i) {
        for (int j=i+1; j<sz; ++j) {
            if (is_similar(A[i], A[j])) {
                // bidirectional edge
                graph[A[i]].push_back(A[j]);
                graph[A[j]].push_back(A[i]);
            }
        }
    }
    // use dfs to find connnected components
    set<string> visited;
    std::function<void(string)> dfs = [&] (string u) {
        visited.insert(u);
        for (auto& v: graph[u]) {
            if (visited.count(v) == 0) {
                dfs(v);
            }
        }
    };
    int ans = 0;
    for (auto& u: A) {
        if (visited.count(u) == 0) {
            dfs(u);
            ans++;
        }
    }
    return ans;
}

}


/*
Given a non-empty array of unique positive integers A, consider the following graph:
There are A.length nodes, labelled A[0] to A[A.length - 1]; There is an edge between A[i] and A[j] if and only if A[i] and A[j] share a common factor greater than 1.
Return the size of the largest connected component in the graph.
Constraints:
    All the values of nums are unique.
Hint: use dfs/disjoint_set to find connected components, and return node count of the largest component
*/
int Solution::largestComponentSize(vector<int>& A) {
if(1) { // disjoint_set solution
    int sz = A.size();
    DisjointSet dsu(sz);
    for (int i=0; i<sz; ++i) {
        for (int j=i+1; j<sz; ++j) {
            if (std::gcd(A[i], A[j]) > 1) { // require c++17
                dsu.unionFunc(i, j);
            }
        }
    }
    int ans = 0;
    map<int, int> groups; // group_id, group_size
    for (int i=0; i<sz; ++i) {
        int grp = dsu.find(i);
        groups[grp]++;
        ans = max(ans, groups[grp]);
    }
    return ans;
}

{ // dfs solution, Time Limit Execeeded
    // build graph
    int sz = A.size();
    map<int, vector<int>> graph;
    for (int i=0; i<sz; ++i) {
        for (int j=i+1; j<sz; ++j) {
            //if (is_similar(A[i], A[j])) {
            if (std::gcd(A[i], A[j]) > 1) { // std would be much faster (6ms vs 350ms)
                graph[A[i]].push_back(A[j]);
                graph[A[j]].push_back(A[i]);
            }
        }
    }
    // use dfs to find a connected component
    set<int> visited;
    // return the number of nodes in the connected component containing node u
    std::function<int(int)> dfs = [&] (int u) {
        visited.insert(u);
        int node_count = 1;
        for (auto v: graph[u]) {
            if (visited.count(v) == 0) {
                node_count += dfs(v);
            }
        }
        return node_count;
    };
    // find all connected components
    int ans = 0;
    for (auto u: A) {
        if (visited.count(u) == 0) {
            ans = max(ans, dfs(u)); 
        }
    }
    return ans;
}

}


/*
Given an array equations of strings that represent relationships between variables, 
each string equations[i] has length 4 and takes one of two different forms: "a==b" or "a!=b".  
Here, a and b are lowercase letters (not necessarily different) that represent one-letter variable names.
Return true if and only if it is possible to assign integers to variables so as to satisfy all the given equations.
Constraints:
    equations[i].length == 4
    equations[i][0] is a lowercase letter.
    equations[i][1] is either '=' or '!'.
    equations[i][2] is '='.
    equations[i][3] is a lowercase letter.
*/
bool Solution::equationsPossible(vector<string>& equations) {
    DisjointSet dsu(128);
    for (const auto& e: equations) {
        // if a==b then put (a, b) into the same group
        if (e[1] == '=') {
            dsu.unionFunc(e[0], e[3]);
        }
    }
    for (const auto& e: equations) {
        // if a!=b then (a, b) must not be in the same group
        if (e[1] == '!') {
            if (dsu.find(e[0]) == dsu.find(e[3])) {
                return false;
            }
        }
    }
    return true;
}


/*
In a network of nodes, each node i is directly connected to another node j if and only if graph[i][j] = 1 (adjacency-matrix representation).

Some nodes `initial` are initially infected by malware. When two nodes are directly connected 
and at least one of those two nodes is infected by malware, both nodes will be infected by malware.  
This spread of malware will continue until no more nodes can be infected in this manner.

Suppose M(initial) is the final number of infected nodes after the spread of malware stops.

We will remove one node from the initial list. Return the node that if removed, would minimize M(initial).
If multiple nodes could be removed to minimize M(initial), return such a node with the smallest index. 

Note that if a node was removed from the initial list of infected nodes, it may still be infected later as a result of the malware spread.      
*/
int Solution::minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
    // build dsu
    int node_count = graph.size();
    DisjointSet dsu(node_count);
    for (int r=0; r<node_count; ++r) {
        for (int c=0; c<node_count; ++c) {
            if (graph[r][c] == 1) {
                dsu.unionFunc(r, c);
            }
        }
    }
    // cluster each group
    map<int, int> group_to_node_cnt_map; // component_id, node_cnt_in_the_component
    for (int i=0; i<node_count; ++i) {
        group_to_node_cnt_map[dsu.find(i)]++;
    }
    // cluster initial into groups
    map<int, int> group_to_initial_node_map; // component_id, node_idx_in_the_component_from_initial
    for (int i=0; i<(int)initial.size(); ++i) {
        group_to_initial_node_map[dsu.find(initial[i])]++;
    }
    int count = 0;
    int idx = INT32_MAX;
    for (int i=0; i<(int)initial.size(); i++) {
        int g = dsu.find(initial[i]);
        if (group_to_initial_node_map[g] == 1) {
            if (group_to_node_cnt_map[g] > count) {
                count = group_to_node_cnt_map[g];
                idx = i;
            }
        }
    }
    return initial[idx == INT32_MAX ? 0 : idx];
}


void calcEquation_scaffold(string equations, string values, string queries, string expectedResult) {
    Solution ss;
    vector<vector<string>> ve = stringTo2DArray<string>(equations);
    vector<double> dv = stringTo1DArray<double>(values);
    vector<vector<string>> vq = stringTo2DArray<string>(queries);
    auto expected = stringTo1DArray<double>(expectedResult);
    auto actual = ss.calcEquation(ve, dv, vq);
    if (actual == expected) {
        SPDLOG_INFO("Case(equations={}, values={}, queries={}, expectedResult={}) passed", equations, values, queries, expectedResult);
    } else {
        SPDLOG_ERROR("Case(equations={}, values={}, queries={}, expectedResult={}) failed, actual={}", equations, values, queries, expectedResult, numberVectorToString(actual));
    }
}


void areSentencesSimilarTwo_scaffold(string s1, string s2, string dict, bool expectedResult) {
    Solution ss;
    vector<string> words1 = stringTo1DArray<string>(s1);
    vector<string> words2 = stringTo1DArray<string>(s2);
    vector<vector<string>> pairs = stringTo2DArray<string>(dict);
    bool actual = ss.areSentencesSimilar(words1, words2, pairs);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", s1, s2, dict, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", s1, s2, dict, expectedResult, actual);
    }
}


void accountsMerge_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<string>> accounts = stringTo2DArray<string>(input);
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> actual = ss.accountsMerge(accounts);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for (const auto& a: actual) {
            std::copy(a.begin(), a.end(), std::ostream_iterator<std::string>(std::cout, ","));
            std::cout << std::endl;
        }   
    }
}


void numSimilarGroups_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<string> accounts = stringTo1DArray<string>(input);
    int actual = ss.numSimilarGroups(accounts);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void largestComponentSize_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> graph = stringTo1DArray<int>(input);
    int actual = ss.largestComponentSize(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void equationsPossible_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<string> equations = stringTo1DArray<string>(input);
    bool actual = ss.equationsPossible(equations);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void countComponents_scaffold(int n, string edges, int expectedResult) {
    Solution ss;
    vector<vector<int>> ve = stringTo2DArray<int>(edges);
    auto actual = ss.countComponents(n, ve);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", n, edges, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", n, edges, expectedResult, actual);
    }
}


void minMalwareSpread_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input1);
    vector<int> initial = stringTo1DArray<int>(input2);
    int actual = ss.minMalwareSpread(graph, initial);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running calcEquation tests:");
    TIMER_START(calcEquation);
    calcEquation_scaffold("[[a,b], [b,c]]", "[2.0, 3.0]", "[[a,c], [b,a], [a,e], [a,a], [x,x]]", "[6.0, 0.5, -1, 1, -1]");
    TIMER_STOP(calcEquation);
    SPDLOG_WARN("calcEquation tests use {} ms", TIMER_MSEC(calcEquation));

    SPDLOG_WARN("Running areSentencesSimilar tests:");
    TIMER_START(areSentencesSimilar);
    areSentencesSimilarTwo_scaffold("[great]", "[great]", "[]", true);
    areSentencesSimilarTwo_scaffold("[great]", "[doubleplus, good]", "[[great, good]]", false);
    areSentencesSimilarTwo_scaffold("[great, acting, skill]", "[fine, drama, talent]", "[[great, good], [fine, good], [acting, drama], [skill, talent]]", true);
    TIMER_STOP(areSentencesSimilar);
    SPDLOG_WARN("areSentencesSimilar tests use {} ms", TIMER_MSEC(areSentencesSimilar));

    SPDLOG_WARN("Running accountsMerge tests:");
    TIMER_START(accountsMerge);
    accountsMerge_scaffold("[[John, johnsmith@mail.com, john00@mail.com],"
                            "[John, johnnybravo@mail.com],"
                            "[John, johnsmith@mail.com, john_newyork@mail.com],"
                            "[Mary, mary@mail.com]]",
                            "[[John, johnnybravo@mail.com],"
                            "[John, john00@mail.com, john_newyork@mail.com, johnsmith@mail.com],"
                            "[Mary, mary@mail.com]]");
    TIMER_STOP(accountsMerge);
    SPDLOG_WARN("accountsMerge tests use {} ms", TIMER_MSEC(accountsMerge));

    SPDLOG_WARN("Running numSimilarGroups tests:");
    TIMER_START(numSimilarGroups);
    numSimilarGroups_scaffold("[star, rats, arts, tars]", 2);
    numSimilarGroups_scaffold("[omv,ovm]", 1);
    TIMER_STOP(numSimilarGroups);
    SPDLOG_WARN("numSimilarGroups tests use {} ms", TIMER_MSEC(numSimilarGroups));

    SPDLOG_WARN("Running largestComponentSize tests:");
    TIMER_START(largestComponentSize);
    largestComponentSize_scaffold("[4,6,15,35]", 4);
    largestComponentSize_scaffold("[20,50,9,63]", 2);
    largestComponentSize_scaffold("[2,3,6,7,4,12,21,39]", 8);
    largestComponentSize_scaffold("[4096,8195,14,24,4122,36,3761,6548,350,54,8249,4155,8252,70,9004,2120,4169,6224,4110,87,6233,4186,6238,4192,1382,4199,104,2153,1845,8310,4231,2185,4245,2212,4261,8359,2222,483,1506,8371,180,2230,6333,2238,2244,197,8391,6353,210,215,216,2265,4315,5872,222,224,8418,6371,4326,234,4331,4332,4334,6384,6387,4342,248,2298,4350,260,8454,2316,270,6419,277,6430,6431,8483,2341,310,311,8521,8525,4432,6483,6487,8541,8542,365,370,4473,396,8596,4501,8598,6551,408,2458,4510,4513,4515,6571,2476,8622,4530,8628,2486,8265,8632,8642,458,2512,4563,4569,4577,763,6629,6642,8700,512,4609,6659,5206,8961,6665,2571,4623,8721,4630,6679,8731,6685,8740,4650,557,8752,8884,4660,4665,4674,2629,4678,2632,4793,2635,6737,4690,2652,2656,2659,8807,6764,628,629,2679,6779,2685,2688,6786,4740,6795,2700,654,4751,5682,6807,678,5575,4785,2738,2739,2740,2743,1140,6844,4800,2754,9675,6853,712,4809,717,8925,8930,6883,4837,8936,8944,4849,754,3162,4853,6906,4859,766,769,812,2832,2834,796,799,8994,6947,4908,813,5619,817,6962,8329,4924,6975,2881,4930,9507,9034,9036,6989,9038,9041,7651,2904,9055,9069,7025,2931,2933,892,7038,7040,4994,2953,2955,7069,9119,5031,9129,5035,7086,1523,9143,9144,7099,5062,3018,9163,7121,9173,7130,3035,6309,3050,5102,3055,1010,1015,7160,7162,5115,5118,8704,7171,9220,9225,7179,7183,1040,7189,9240,9252,3112,7215,5171,5173,7226,9280,5186,7240,9295,5200,5201,5202,3158,9306,7268,1131,7276,5232,9332,1151,9345,1154,7303,7309,1182,7327,1184,1190,7337,7341,1199,9396,7353,9404,3275,7370,1227,7374,3284,3289,3299,5348,5349,1255,1266,5371,5376,9481,7438,8408,7444,7726,3355,1313,6363,3371,3373,5425,9527,9531,7486,3394,5443,5444,7494,9547,5454,9569,3429,5478,7533,3445,9591,4330,1406,1600,3468,3471,5527,7581,7591,9642,5557,9657,7071,7614,7618,3523,9668,1479,5576,7627,9676,9680,7639,9698,5603,1513,2642,9712,7763,5621,7081,9722,9728,9803,9733,3592,1546,5038,9753,1582,7730,3637,5696,9794,1609,951,7756,7758,5711,1617,9811,7764,1632,5738,9836,1647,3703,5754,9851,1661,5397,2668,9867,3728,9874,7827,1684,7834,1693,3747,3748,7856,9905,3766,1728,5828,2308,1735,7881,5834,7887,1745,6435,7896,1754,9950,8485,1760,9954,3822,3824,1777,1782,981,5891,7942,7946,8168,7958,5912,4834,5915,7973,5930,5941,5943,3899,5951,8005,8006,8161,1864,8013,3919,3926,8024,8038,6120,6004,6013,6015,6026,1933,6031,8081,3990,1944,1947,1952,6055,1963,8111,4024,6074,6075,5451,1997,2002,4053,8152,4057,4059,4063,2016,6113,4072,8179,4089]", 432);
    TIMER_STOP(largestComponentSize);
    SPDLOG_WARN("largestComponentSize tests use {} ms", TIMER_MSEC(largestComponentSize));

    SPDLOG_WARN("Running equationsPossible tests:");
    TIMER_START(equationsPossible);
    equationsPossible_scaffold("[a==b, b!=a]", false);
    equationsPossible_scaffold("[a==b, b==a]", true);
    equationsPossible_scaffold("[a==b, b==c, a==c]", true);
    equationsPossible_scaffold("[a==b, b!=c, a==c]", false);
    equationsPossible_scaffold("[a==b, c!=b, a==c]", false);
    equationsPossible_scaffold("[a==a, b==d, x!=z]", true);
    equationsPossible_scaffold("[a!=b, b!=c, c!=a]", true);
    TIMER_STOP(equationsPossible);
    SPDLOG_WARN("equationsPossible tests use {} ms", TIMER_MSEC(equationsPossible));

    SPDLOG_WARN("Running countComponents tests:");
    TIMER_START(countComponents);
    countComponents_scaffold(5, "[[0,1],[1,2],[3,4]]", 2);
    countComponents_scaffold(5, "[[0,1],[1,2],[2,3],[3,4]]", 1);
    TIMER_STOP(countComponents);
    SPDLOG_WARN("countComponents tests use {} ms", TIMER_MSEC(countComponents));

    SPDLOG_WARN("Running minMalwareSpread tests:");
    TIMER_START(minMalwareSpread);
    minMalwareSpread_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", "[0,1]", 0);
    minMalwareSpread_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", "[0,1,2]", 2);
    minMalwareSpread_scaffold("[[1,0,0],[0,1,0],[0,0,1]]", "[0,2]", 0);
    minMalwareSpread_scaffold("[[1,1,1],[1,1,1],[1,1,1]]", "[1,2]", 1);
    minMalwareSpread_scaffold("[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,1,0],[0,0,0,1,0,0],[0,0,1,0,1,0],[0,0,0,0,0,1]]", "[4,3]", 4);
    TIMER_STOP(minMalwareSpread);
    SPDLOG_WARN("minMalwareSpread tests use {} ms", TIMER_MSEC(minMalwareSpread));
}
