#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

extern "C" {

/**
 * Segment Tree for efficient range queries and updates
 * Supports: range max/min queries, top-k queries, point updates
 */
class SegmentTree {
private:
    std::vector<double> tree;
    std::vector<int> indices;
    int n;
    
    void build(const std::vector<double>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2*node, start, mid);
            build(arr, 2*node+1, mid+1, end);
            tree[node] = std::max(tree[2*node], tree[2*node+1]);
        }
    }
    
    void update_helper(int node, int start, int end, int idx, double val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update_helper(2*node, start, mid, idx, val);
            } else {
                update_helper(2*node+1, mid+1, end, idx, val);
            }
            tree[node] = std::max(tree[2*node], tree[2*node+1]);
        }
    }
    
    double query_helper(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return -std::numeric_limits<double>::infinity();
        }
        if (l <= start && end <= r) {
            return tree[node];
        }
        int mid = (start + end) / 2;
        double p1 = query_helper(2*node, start, mid, l, r);
        double p2 = query_helper(2*node+1, mid+1, end, l, r);
        return std::max(p1, p2);
    }

public:
    SegmentTree(int size) : n(size) {
        tree.resize(4 * n, 0.0);
        indices.resize(n);
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
    }
    
    void update(int idx, double val) {
        if (idx < 0 || idx >= n) return;
        update_helper(1, 0, n-1, idx, val);
    }
    
    double query_range(int l, int r) {
        if (l < 0 || r >= n || l > r) return 0.0;
        return query_helper(1, 0, n-1, l, r);
    }
    
    std::vector<int> get_top_k(int k, std::vector<double>& values) {
        std::vector<std::pair<double, int>> pairs;
        for (int i = 0; i < values.size() && i < n; i++) {
            pairs.push_back({values[i], i});
        }
        
        std::sort(pairs.begin(), pairs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::vector<int> result;
        for (int i = 0; i < std::min(k, (int)pairs.size()); i++) {
            result.push_back(pairs[i].second);
        }
        return result;
    }
    
    std::vector<double>& get_tree() {
        return tree;
    }
};

// C API for Python ctypes
struct SegmentTreeHandle {
    SegmentTree* tree;
    std::vector<double> values;
    std::vector<int> top_k_result;
};

void* create_segment_tree(int size) {
    SegmentTreeHandle* handle = new SegmentTreeHandle();
    handle->tree = new SegmentTree(size);
    handle->values.resize(size, 0.0);
    std::cout << "[C++] Created Segment Tree with " << size << " nodes" << std::endl;
    return handle;
}

void update_value(void* handle_ptr, int index, double value) {
    SegmentTreeHandle* handle = static_cast<SegmentTreeHandle*>(handle_ptr);
    if (handle && handle->tree) {
        handle->tree->update(index, value);
        if (index < handle->values.size()) {
            handle->values[index] = value;
        }
        std::cout << "[C++] Updated index " << index << " with value " << value << std::endl;
    }
}

double query_range_max(void* handle_ptr, int left, int right) {
    SegmentTreeHandle* handle = static_cast<SegmentTreeHandle*>(handle_ptr);
    if (handle && handle->tree) {
        double result = handle->tree->query_range(left, right);
        std::cout << "[C++] Range query [" << left << ", " << right << "] = " << result << std::endl;
        return result;
    }
    return 0.0;
}

int* query_top_k(void* handle_ptr, int k, int size) {
    SegmentTreeHandle* handle = static_cast<SegmentTreeHandle*>(handle_ptr);
    if (handle && handle->tree) {
        handle->top_k_result = handle->tree->get_top_k(k, handle->values);
        std::cout << "[C++] Top-" << k << " query completed" << std::endl;
        return handle->top_k_result.data();
    }
    return nullptr;
}

void destroy_segment_tree(void* handle_ptr) {
    SegmentTreeHandle* handle = static_cast<SegmentTreeHandle*>(handle_ptr);
    if (handle) {
        if (handle->tree) {
            delete handle->tree;
        }
        delete handle;
        std::cout << "[C++] Segment Tree destroyed" << std::endl;
    }
}

double get_value(void* handle_ptr, int index) {
    SegmentTreeHandle* handle = static_cast<SegmentTreeHandle*>(handle_ptr);
    if (handle && index < handle->values.size()) {
        return handle->values[index];
    }
    return 0.0;
}

} // extern "C"