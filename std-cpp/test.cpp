#include <iostream>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

using namespace std;


int main() {
	int count = 0;
	std::string word;
	while (std::cin >> word) {
		std::cout << count++ << ": " << word << std::endl;
	}
}
