#include "leetcode.h"

using namespace std;

static const int LOOP_COUNT = 100;

void binarySearchTester(int arraySize);
void lowerBoundSearchTester(int arraySize);
void upperBoundSearchTester(int arraySize);

int main(int argc, char* argv[]) {
    int arraySize = 0;
    int testType = 0;
    string path(argv[0]);
    string programName = path.substr(path.find_last_of('/')+1);
    if (argc != 3) {
        cout << "Usage: " << programName << " ArraySize" << " TestType\n" ;
        cout << "\tArraySize must be a positive integer\n";
        cout << "\tTestType=0 test all\n";
        cout << "\tTestType=1 binary search test\n";
        cout << "\tTestType=2 lower bound search test\n";
        cout << "\tTestType=3 upper bound search test\n";
        return 1;
    } else {
        arraySize = atoi(argv[1]);
        testType = atoi(argv[2]);
        if (arraySize <= 0) {
            cout << "ArraySize must be a positive integer!\n";
            return 1;
        } else if (testType<0 || testType>3) {
            cout << "TestType must be choosen from 0,1,2,3\n";
            return 1;
        }
    }

    srand(1234);

    if (testType == 1) {
        binarySearchTester(arraySize);
    } else if (testType == 2) {
        lowerBoundSearchTester(arraySize);
    } else if (testType == 3) {
        upperBoundSearchTester(arraySize);
    } else {
        binarySearchTester(arraySize);
        lowerBoundSearchTester(arraySize);
        upperBoundSearchTester(arraySize);
    }

    return 0;
}

void binarySearchTester(int arraySize) {
    auto worker = [&] (const vector<int>& input, int key) {
        bool found = false;
        int l = 0; // l is inclusive
        int r = input.size() - 1; // r is inclusive
        while (l <= r) {
            int m = (l+r)/2;
            if (input[m] == key) { // found key
                found = true;
                break;
            } else if (input[m] < key) { // key must be in right partition if it exists in input
                l = m+1;
            } else { // otherwise in left partition
                r = m-1;
            }
        }
        bool expectedResult = std::binary_search(input.begin(), input.end(), key);
        if (found != expectedResult) {
            SPDLOG_ERROR("binarySearchTester failed, arraySize: {}, expected result: {}, acutal result: {}", arraySize, expectedResult, found);
            abort();
        }
    };

    SPDLOG_WARN("Running binarySearchTester tests");
    TIMER_START(binarySearchTester);
    vector<int> input;
    generateTestArray(input, arraySize, false, true);
    for (int i = 0; i < LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    generateTestArray(input, arraySize, true, true);
    for (int i = 0; i < LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    TIMER_STOP(binarySearchTester);
    SPDLOG_WARN("binarySearchTester tests use {} ms", TIMER_MSEC(binarySearchTester));
}


// return the first element that is greater or equal to key, array must be sorted in advance
void lowerBoundSearchTester(int arraySize) {
    auto worker = [&](const vector<int>& input, int key) {
        int l = 0; // l is inclusive
        int r = input.size(); // r is not inclusive
        while (l < r) {
            int m = (l+r)/2;
            if (input[m] < key) {
                l = m+1;
            } else { // when input[m] == key, we move r to left to find the leftmost element is greater than or equal to key
                r = m;
            }
        }
        auto it = std::lower_bound(input.begin(), input.end(), key);
        int expectedResult = std::distance(input.begin(), it);
        if (l != expectedResult) {
            SPDLOG_ERROR("lowerBoundSearchTester failed, arraySize: {}, key: {}, expected result: {}, acutal result: {}, {}\n", arraySize, key, expectedResult, l, r);
            abort();
        }
    };

    SPDLOG_WARN("Running lowerBoundSearchTester tests");
    TIMER_START(lowerBoundSearchTester);
    vector<int> input;
    generateTestArray(input, arraySize, false, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    generateTestArray(input, arraySize, true, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    TIMER_STOP(lowerBoundSearchTester);
    SPDLOG_WARN("lowerBoundSearchTester tests use {} ms", TIMER_MSEC(lowerBoundSearchTester));
}


// return the first element that is greater than key, array must be sorted in advance
void upperBoundSearchTester(int arraySize) {
    auto worker = [&](const vector<int>& input, int key) {
        int l=0; // l is inclusive
        int r = input.size(); // r is not inclusive
        while (l < r) {
            int m = (l+r)/2;
            if (input[m] <= key) { // when input[m] == key, we move l to right to find the leftmost element is greater than key
                l = m+1;
            } else {
                r = m;
            }
        }
        auto it = std::upper_bound(input.begin(), input.end(), key);
        int expectedResult = std::distance(input.begin(), it);
        if (l != expectedResult) {
            SPDLOG_ERROR("upperBoundSearchTester failed, arraySize: {}, key: {}, expected result: {}, acutal result: {}, {}\n", arraySize, key, expectedResult, l, r);
            abort();
        }
    };

    SPDLOG_WARN("Running upperBoundSearchTester tests");
    TIMER_START(upperBoundSearchTester);
    vector<int> input;
    generateTestArray(input, arraySize, false, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    generateTestArray(input, arraySize, true, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    TIMER_STOP(upperBoundSearchTester);
    SPDLOG_WARN("upperBoundSearchTester tests use {} ms", TIMER_MSEC(upperBoundSearchTester));
}