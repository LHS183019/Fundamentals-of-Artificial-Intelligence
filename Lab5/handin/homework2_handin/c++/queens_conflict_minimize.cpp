#include <cmath>
#include <ctime>
#include <chrono>
#include <string>

#include "problem/queens_constraint.hpp"
#include "algorithm/conflicts_minimize.hpp"
#include "utils/selection.hpp"

// 在保证非负单调递增的前提下，估值函数仅对轮盘赌算法起作用
double alpha = 10;
double value_of(int value){
    return exp(alpha * value);
}

int main(int argc, char* argv[]){
    int n = 1000;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }
    
    try {
        n = std::stoi(argv[1]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << argv[1] << " is not a valid integer." << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << argv[1] << " is out of range." << std::endl;
        return 1;
    }

    // 用于统计耗时
    auto start = std::chrono::high_resolution_clock::now();

    std::ios::sync_with_stdio(false);

    
    QueensConstraintSatisfaction q(n);
    
    ConflictMinimize<QueensConstraintSatisfaction> cm(q);
    
    FirstBetterSelection fbs;
    RouletteSelection rs;
    MaxSelection ms;

    // 随机重启尝试10轮，每轮最多更改变元4n次
    // ms: 最大选择算法，优先选择冲突最多的变元更改，优先更改到冲突最小的值（移步algorithm/conflict_minimize.hpp阅读）
    // value_of: 因为使用最大选择算法，因此估值函数直接使用默认的指数函数即可
    cm.search(50, n, ms);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "queens count: "<< n << std::endl;     
    std::cout << "Total time: " << duration.count() << "ms" << std::endl;

    system("pause");
    return 0;
}