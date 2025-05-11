#include "disjoint_set.h"
#include <stack>

using namespace std;

DisjointSet::DisjointSet(int n) {
    m_rank.resize(n+1, 0);  // rank of node i. the upperbound of the height of node i
    m_parent.resize(n+1, 0); // parent of node i
    // initialization: set parent of each node to itself
    std::iota(m_parent.begin(), m_parent.end(), 0);
}

int DisjointSet::find(int x) {
#if defined(USE_RECURSIVE_SOLUTION)
    return find_recursive(x);
#else
    return find_iterative(x);
#endif
}

int DisjointSet::find_recursive(int x) {
    if (m_parent[x] != x) { // Be cautious, there must be a `if` clause instead of `while`
        m_parent[x] = find_recursive(m_parent[x]);
    }
    return m_parent[x];
}

int DisjointSet::find_iterative(int x) {
    stack<int> s;
    while (x != m_parent[x]) {
        s.push(x);
        x = m_parent[x];
    }
    // assert(x == m_parent[x]);
    while (!s.empty()) {
        m_parent[s.top()] = x;
        s.pop();
    }
    return m_parent[x];
}

bool DisjointSet::unionFunc(int x, int y) {
    // 1. find the parent of node x, y
    int px = find(x);
    int py = find(y);
    if(px == py) {
        return false; // cycle detected
    }
    // 2. set parent of node with lower rank point to the parent of node with higher rank 
    if (m_rank[px] > m_rank[py]) {
        m_parent[py] = px; 
    } else if (m_rank[px] < m_rank[py]) {
        m_parent[px] = py;
    } else { // for ties, increase the rank of either node
        m_parent[px] = py;
        ++m_rank[py];
    }
    return true;
}
