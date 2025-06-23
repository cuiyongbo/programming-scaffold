#include "leetcode.h"

using namespace std;

/* leetcode exercise: 912 */

enum AlgorithmType {
    AlgorithmType_none,
    AlgorithmType_quickSort,
    AlgorithmType_countingSort,
    AlgorithmType_radixSort,
    AlgorithmType_heapSort,
    AlgorithmType_mergeSort,
    AlgorithmType_binarySearchTree,
    AlgorithmType_insertionSort,
};

const char* AlgorithmType_toString(AlgorithmType type) {
    const char* str = nullptr;
    switch (type) {
    case AlgorithmType::AlgorithmType_quickSort:
        str = "quickSort";
        break;
    case AlgorithmType::AlgorithmType_countingSort:
        str = "countingSort";
        break;
    case AlgorithmType::AlgorithmType_radixSort:
        str = "radixSort";
        break;
    case AlgorithmType::AlgorithmType_heapSort:
        str = "heapSort";
        break;
    case AlgorithmType::AlgorithmType_mergeSort:
        str = "mergeSort";
        break;
    case AlgorithmType::AlgorithmType_binarySearchTree:
        str = "binarySearchTree";
        break;
    case AlgorithmType::AlgorithmType_insertionSort:
        str = "insertionSort";
        break;
    default:
        str = "unknown";
        break;
    }
    return str;
}

class Solution {
public:
    void sortArray(vector<int>& nums, AlgorithmType type=AlgorithmType_none); 
    void quickSort(vector<int>& nums);
    void heapSort(vector<int>& nums);
    void mergeSort(vector<int>& nums);
    void bstSort(vector<int>& nums);
    void insertionSort(vector<int>& nums);
    void countingSort(vector<int>& nums);
    void radixSort(vector<int>& nums);

private:
    void quick_sort_worker(vector<int>& nums, int l, int r);
    int quick_sort_partitioner(vector<int>& nums, int l, int r);
};

void Solution::sortArray(vector<int>& nums, AlgorithmType type) {
    if (type == AlgorithmType_quickSort) {
        quickSort(nums);
    } else if (type == AlgorithmType_countingSort) {
        countingSort(nums);
    } else if (type == AlgorithmType_heapSort) {
        heapSort(nums);
    } else if (type == AlgorithmType_mergeSort) {
        mergeSort(nums);
    } else if (type == AlgorithmType_binarySearchTree) {
        bstSort(nums);
    } else if (type == AlgorithmType_insertionSort) {
        insertionSort(nums);
    } else if (type == AlgorithmType_radixSort) {
        radixSort(nums);
    } else {
        cout << "invalid algorithm type" << endl;
    }
}

/*
runtime complexity O(nlogn) on average case
1. pick and element, called a pivot, from the array
2. partitioning: reorder the array so that the elements with values less than the pivot come before the pivot, and the elements with values greater than the pivot come after it (equal values can go either way). after the partition, the pivot is in its final position
3. recursively apply the above steps to all sub-arrays
*/
void Solution::quickSort(vector<int>& nums) {
    // l, r are inclusive
    auto naive_partitioner = [&] (int l, int r) {
        // randomly choose pivots in case that the algorithm degrades when the array is already sorted or nearly sorted
        // it may accelerate the algorithm by 10 times
        int rr = rand()%(r-l+1) + l;
        swap(nums[rr], nums[r]);
        int i = l-1;
        int pivot = nums[r];
        for (int k=l; k<r; ++k) {
            // put elements less than pivot to nums[l:i]
            if (nums[k] < pivot) { // num[k]<pivot, l <= k <= i
                ++i;
                std::swap(nums[i], nums[k]);
            }
        }
        // move pivot to its final sorted position
        ++i;
        std::swap(nums[i], nums[r]);
        return i;
    };
    // l, r are inclusive
    std::function<void(int, int)> dac = [&] (int l, int r) {
        if (l >= r) { // trivial case
            return;
        }
        int m = naive_partitioner(l, r);
        // nums[m] is in its final sorted position after parititon
        // continue to sort remaining array
        dac(l, m-1);
        dac(m+1, r);
    };
    dac(0, nums.size()-1);
}


/* 
\Theta(k) [k = max(array) - min(array)]
NOT suitable for sparse arrays whose elements split in a large domain,
such as [INT_MIN, INT_MAX], suffering to range overflow
applications:
    - sortColors
*/
void Solution::countingSort(vector<int>& nums) {
    // 1. find the range of the elements
    auto p = minmax_element(nums.begin(), nums.end());
    int l = *(p.first);
    int r = *(p.second);
    long range = r-l+1; // we have to use long type to store range in case of range overflow
    SPDLOG_DEBUG("CountingSort(min={}, max={}, range={})", l, r, range);
    // 2. we use the value of array elements as indexing and counting
    vector<int> counting(range, 0);
    for (auto n: nums) {
        counting[n-l]++;
    }
    // 3. reinsert elements according to counting array
    nums.clear();
    for (long i=0; i<range; ++i) {
        if (counting[i] != 0) {
            // std::vector<int>::iterator std::vector<int>::insert(std::vector<int>::const_iterator __position, std::size_t __n, const int &__x)
            // This function will insert a specified number of copies of the given data before the location specified by position.
            nums.insert(nums.end(), counting[i], long(i+l));
        }
    }
}


/*
// \Theta(log_b(k)*(n+k)), k is the maximum element value in the element

Radix-Sort(A, d)
    for i=1 to d
        use a stable sort to sort array A on i_th digit

for example, given an input array: [2457, 2400, 7939, 7705, 6845, 8773, 8469, 3366, 3604, 6218]

loop 1: (2400), (8773), (3604), (7705, 6845), (3366), (2457), (6218), (7939, 8469)
loop 2: (2400, 3604, 7705), (6218), (7939), (6845), (2457), (3366, 8469), (8773)
loop 3: (6218), (3366), (2400, 2457, 8469), (3604), (7705, 8773), (6845), (7939)
loop 4:  (2400, 2457), (3366, 3604), (6218, 6845), (7705, 7939), (8469, 8773)
*/
void Solution::radixSort(vector<int>& nums) {
    // calculate input value range
    auto p = minmax_element(nums.begin(), nums.end());
    int l = *(p.first);
    int r = *(p.second);
    // map array elements to [0, r-l], NON-NEGATIVE
    std::transform(nums.begin(), nums.end(), nums.begin(), [&](int a) {return a-l;}); // watch out! nums[i] may overflow during transforming

    /*
    # ./array_sort 10000
    [2025-04-26 11:54:30.525] [warning] [array_sort.cpp:435] Running batch tests(array_size=10000)
    [2025-04-26 11:54:30.525] [info] [array_sort.cpp:358] Running countingSort tests
    [2025-04-26 11:54:30.540] [info] [array_sort.cpp:381] countingSort tests use 15.3112 ms
    [2025-04-26 11:54:30.540] [info] [array_sort.cpp:358] Running radixSort tests
    [2025-04-26 11:54:30.631] [info] [array_sort.cpp:381] radixSort tests use 91.126711 ms
    [2025-04-26 11:54:30.631] [info] [array_sort.cpp:358] Running insertionSort tests
    [2025-04-26 11:54:31.586] [info] [array_sort.cpp:381] insertionSort tests use 954.826413 ms
    [2025-04-26 11:54:31.586] [info] [array_sort.cpp:358] Running mergeSort tests
    [2025-04-26 11:54:31.619] [info] [array_sort.cpp:381] mergeSort tests use 32.868097 ms
    [2025-04-26 11:54:31.619] [info] [array_sort.cpp:358] Running quickSort tests
    [2025-04-26 11:54:31.653] [info] [array_sort.cpp:381] quickSort tests use 33.590679 ms
    [2025-04-26 11:54:31.653] [info] [array_sort.cpp:358] Running heapSort tests
    [2025-04-26 11:54:31.694] [info] [array_sort.cpp:381] heapSort tests use 41.260992 ms
    [2025-04-26 11:54:31.694] [warning] [array_sort.cpp:444] batch tests using 1169.08 milliseconds
    # ./array_sort 100000
    [2025-04-26 11:54:39.132] [warning] [array_sort.cpp:435] Running batch tests(array_size=100000)
    [2025-04-26 11:54:39.132] [info] [array_sort.cpp:358] Running countingSort tests
    [2025-04-26 11:54:39.223] [info] [array_sort.cpp:381] countingSort tests use 90.644472 ms
    [2025-04-26 11:54:39.223] [info] [array_sort.cpp:358] Running radixSort tests
    [2025-04-26 11:54:40.200] [info] [array_sort.cpp:381] radixSort tests use 976.858026 ms
    [2025-04-26 11:54:40.200] [info] [array_sort.cpp:358] Running insertionSort tests
    [2025-04-26 11:56:13.234] [info] [array_sort.cpp:381] insertionSort tests use 93034.092811 ms
    [2025-04-26 11:56:13.234] [info] [array_sort.cpp:358] Running mergeSort tests
    [2025-04-26 11:56:13.602] [info] [array_sort.cpp:381] mergeSort tests use 368.23889199999996 ms
    [2025-04-26 11:56:13.602] [info] [array_sort.cpp:358] Running quickSort tests
    [2025-04-26 11:56:13.994] [info] [array_sort.cpp:381] quickSort tests use 391.669153 ms
    [2025-04-26 11:56:13.994] [info] [array_sort.cpp:358] Running heapSort tests
    [2025-04-26 11:56:14.470] [info] [array_sort.cpp:381] heapSort tests use 475.754686 ms
    [2025-04-26 11:56:14.470] [warning] [array_sort.cpp:444] batch tests using 95337.40 milliseconds
    */
    auto sort_by_counting_sort = [&] (int base) {
        int sz = nums.size();
        vector<vector<int>> counting(10); // digit: element values
        for (int i=0; i<sz; i++) {
            int ni = nums[i]/base%10; // this operation takes time
            counting[ni].push_back(nums[i]);
        }
        nums.clear();
        for (const auto& p: counting) {
            nums.insert(nums.end(), p.begin(), p.end());
        }
    };
    /*
    in naive implementation, the runtime of radixSort is about 40 times of insertionSort
    # ./array_sort 1000
    [2025-04-26 11:31:32.155] [info] [array_sort.cpp:334] Running radixSort tests
    [2025-04-26 11:31:32.558] [info] [array_sort.cpp:357] radixSort tests use 402.431399 ms
    [2025-04-26 11:31:32.558] [info] [array_sort.cpp:334] Running insertionSort tests
    [2025-04-26 11:31:32.570] [info] [array_sort.cpp:357] insertionSort tests use 12.235455 ms
    */
   /*
    // workhorse
    auto sort_by_ith_digit = [&] (int base) {
        int sz = nums.size();
        for (int i=0; i<sz; i++) {
            int tmp = nums[i]; // save nums[i]
            int ni = nums[i]/base%10; // this operation takes time
            for (int j=0; j<i; j++) {
                int nj = nums[j]/base%10;
                if (ni<nj) {
                    // we should put ni at nj, but before we do that, we must shift nums[j:i-1] to right by one place to save place for nums[j]
                    for (int k=i; k>j; k--) {
                        nums[k] = nums[k-1];
                    }
                    nums[j] = tmp; // put nums[i] to nums[j]
                    break;
                }
            }
        }
    };
   */
    // main program
    int base = 1;
    int max_num = r-l;
    while (max_num > 0) {
        // perform stable sort on ith digit, such as insertion sort
        //sort_by_ith_digit(base);
        sort_by_counting_sort(base);
        base *= 10;
        max_num /= 10;
    }
    // remap array elements to original value
    std::transform(nums.begin(), nums.end(), nums.begin(), [&](int a) {return a+l;});
    return;
}


// O(nlogn) for worst-case running time
void Solution::heapSort(vector<int>& nums) {
if (1) {
    // end is not inclusive
    auto sift_down = [&] (int l, int end) {
        int root = l;
        while (root < end) {
            int left = 2*root+1;
            int right = 2*root+2;
            int max_i = root;
            if (left<end && nums[left] > nums[max_i]) {
                max_i = left;
            }
            if (right<end && nums[right] > nums[max_i]) {
                max_i = right;
            }
            // the heap is in its shape
            if (max_i == root) {
                break;
            }
            swap(nums[root], nums[max_i]);
            root = max_i;
        }
    };
    // 1. build heap with sift-up
    int size = nums.size();
    for (int i=size/2; i>=0; i--) {
        sift_down(i, size);
    }
    // 2. extract the largest element from the remaining array one by one
    for (int i=size-1; i>0; i--) {
        std::swap(nums[0], nums[i]);
        // restore the heap property for nums[0:i] (note that i is not inclusive) with sift-down
        sift_down(0, i);
    }
    return;
}

{ // std solution
    // in `std::priority_queue` template, we perform `compare(child, root)` test to see whether root, left-child, right-child are in heap-order or not
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap(nums.begin(), nums.end());
    nums.clear();
    while (!min_heap.empty()) {
        nums.push_back(min_heap.top()); min_heap.pop();
    }
    return;
}

}


/*
// O(nlogn) for worst-case running time
1. divide the unsorted list into n sublists, each containing 1 element
2. repeatedly merge sublists to produce new sorted sublists untill there is only 1 sublist remaining. this will be the sorted list
*/
void Solution::mergeSort(std::vector<int>& nums) {
    std::vector<int> twins = nums;
    // l, r are inclusive
    std::function<void(int, int)> dac = [&] (int l, int r) {
        if (l >= r) { // trivial case
            return;
        }
        // std::vector<int> twins = nums;
        int m = (l+r)/2;
        // divide and conquer
        dac(l, m); dac(m+1, r);
        // merge
        int i=l;
        int j=m+1;
        for (int k=l; k<=r; k++) {
            if ((i<=m && nums[i]<nums[j]) || (j>r)) {
                twins[k] = nums[i++];
            } else {
                twins[k] = nums[j++];
            }
        }
        // swap elements in [l, r]. NOTE that dac(l, r) only operates on nums[l:r], you cannot call `nums=twins;`, which will break the work done by other routines
        std::copy(twins.begin()+l, twins.begin()+r+1, nums.begin()+l);
        // don't call `swap(twins, nums);` here, it will overwrite work by other subroutine unless you create a new twins for every subroutine
    };
    return dac(0, nums.size()-1);
}


// \Theta(nlogn)
void Solution::bstSort(vector<int>& nums) {
    // `std::multiset` maybe implemented with red-black tree
    // the hash of int type is itself. refer to [Why std::hash<int> seems to be identity function](https://stackoverflow.com/questions/38304877/why-stdhashint-seems-to-be-identity-function)
    std::multiset<int> s(nums.begin(), nums.end());
    std::copy(s.begin(), s.end(), nums.begin());
}


// \Theta(n^2)
/*
    [l, r), i
    [l, i] is sorted in ascending order
    [i+1, r) is not sorted
    the algorithm consists of two loops:
        loop one: expand the sorted array to right by one at each iteration
        loop two: make sure the left part is sorted
*/
void Solution::insertionSort(vector<int>& nums) {
    int l = 0;
    int r = nums.size(); // r is not inclusive
    for (int i=l; i<r; i++) {
        int p = nums[i]; // we must save nums[i] for later assignment
        for (int j=l; j<i; j++) {
            // find the first index j, at which nums[j] > p, meaning nums[i] should be inserted before nums[j]
            // we use (nums[j] <= p) test to make insertionSort to be a stable sort
            if (nums[j] <= p) {
                continue;
            }
            // p should be placed at nums[j]
            // but before doing that we need shift nums[j, i-1] to right by one
            for (int k=i; k>j; k--) { // watch out, nums[i] has been replaced
                nums[k] = nums[k-1];
            }
            nums[j] = p;
            break;
        }
    }
    return;
}


void sortArray_scaffold(string input, AlgorithmType type) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    ss.sortArray(vi, type);
    if(std::is_sorted(vi.begin(), vi.end())) {
        SPDLOG_INFO("case({}, {}) passed", input, AlgorithmType_toString(type));
    } else {
        SPDLOG_INFO("case({}, {}) failed", input, AlgorithmType_toString(type));
    }
}


void batch_test_scaffold(int array_scale, AlgorithmType type) {
    SPDLOG_INFO("Running {} tests", AlgorithmType_toString(type));
    TIMER_START(batch_test_scaffold);
    std::random_device rd;
    std::mt19937 g(rd());
    Solution ss;
    vector<int> vi; vi.reserve(array_scale);
    for (int i=0; i<array_scale; i++) {
        if (type == AlgorithmType_countingSort) {
            vi.push_back(rand()%100000);
        } else {
            vi.push_back(rand());
        }
    }
    for (int i=0; i<100; ++i) {
        //SPDLOG_INFO("Running {} tests at {}", AlgorithmType_toString(type), i+1);
        int n = rand() % array_scale;
        std::shuffle(vi.begin(), vi.begin()+n, g);
        ss.sortArray(vi, type);
        if(!std::is_sorted(vi.begin(), vi.end())) {
            SPDLOG_ERROR("Case(array_scale={}, array_size={}, algorithm={}) failed", array_scale, n, AlgorithmType_toString(type));
        }
    }
    TIMER_STOP(batch_test_scaffold);
    SPDLOG_INFO("{} tests use {} ms", AlgorithmType_toString(type), TIMER_MSEC(batch_test_scaffold));
}


void basic_test() {
    SPDLOG_INFO("Running sortArray tests:");
    TIMER_START(sortArray);
    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_quickSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_quickSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_quickSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_countingSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_countingSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_countingSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_heapSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_heapSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_heapSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_mergeSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_mergeSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_mergeSort);
    sortArray_scaffold("[3,1,2]", AlgorithmType::AlgorithmType_mergeSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_binarySearchTree);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_binarySearchTree);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_binarySearchTree);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_insertionSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_insertionSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_insertionSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_radixSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_radixSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_radixSort);

    TIMER_STOP(sortArray);
    SPDLOG_INFO("sortArray using {:.2f} milliseconds", TIMER_MSEC(sortArray));
}


int main(int argc, char* argv[]) {
    //basic_test();

    int array_size = 1000;
    if (argc > 1) {
        array_size = std::atoi(argv[1]);
        if (array_size <= 0) {
            printf("Usage: %s [arrary_size]", argv[0]);
            printf("\tarrary_size must be positive, default to 100 if unspecified");
            return -1;
        }
    }

    SPDLOG_WARN("Running batch tests(array_size={})", array_size);
    TIMER_START(sortArray_batch_test);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_countingSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_radixSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_insertionSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_mergeSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_quickSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_heapSort);
    TIMER_STOP(sortArray_batch_test);
    SPDLOG_WARN("batch tests using {:.2f} milliseconds", TIMER_MSEC(sortArray_batch_test));
}
