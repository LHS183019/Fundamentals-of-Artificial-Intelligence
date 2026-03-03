#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <stdexcept>
#include <chrono>

#include "interface/state_local.hpp"
#include "problem/queens_move.hpp"
#include "problem/queens_swap.hpp"
#include "algorithm/hill_climb.hpp"
#include "utils/selection.hpp"

double log_n;
int n;
// 使用RouletteSelection时可以尝试修改下面的两个估值函数来获得更好的效果
double queens_move_state_value_estimator(const QueensMoveState& state){
    return exp(-state.conflicts());    
}

double queens_swap_state_value_estimator(const QueensSwapState& state){
    return exp(-log_n * state.state().conflicts());    
}

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }
    
    try {
        n = std::stoi(argv[1]);
        log_n = log(n);
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

    // 选择第一个比当前状态好的邻居状态
    FirstBetterSelection f_selection;
    // 按照与价值成正比的概率选择邻居状态
    RouletteSelection r_selection;
    // 选择价值最高的邻居状态
    MaxSelection m_selection;

    // QueensMoveState state(n);
    
    // // 用QueensMoveState的问题建模（动作为将某一行的皇后移动到某一列）
    // HillClimb<QueensMoveState> hcs(state);
    
    // // 适应度达到1.0（无冲突皇后，由估值函数决定）则算法终止，至多迭代4n步，随机重启运行5次。
    
    // hcs.search(queens_move_state_value_estimator, 1.0, n << 2, r_selection, 5);
    
    // 用QueensSwapState的问题建模（动作为交换两行的皇后）
    QueensSwapState state(n);
    HillClimb<QueensSwapState> hcs(state);
    hcs.search(queens_swap_state_value_estimator, 1.0, n/2, f_selection, 10);
    
    // 以毫秒计算时间差
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "queens count: "<< n << std::endl;
    std::cout << "Total time: " << duration.count() << "ms" << std::endl;

    return 0;
}
