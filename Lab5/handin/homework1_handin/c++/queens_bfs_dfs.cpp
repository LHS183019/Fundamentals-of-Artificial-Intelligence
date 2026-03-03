#include <iostream>
#include <ctime>
#include <string>
#include "problem/queens.hpp"

#include "algorithm/depth_first_search.hpp"
#include "algorithm/breadth_first_search.hpp"

int main(int argc, char* argv[]){
    std::ios::sync_with_stdio(false);

    time_t t0 = time(nullptr);
    
    int numQueens = (argc > 1) ? std::stoi(argv[1]) : 12;
    QueensState state(numQueens);
    BreadthFirstSearch<QueensState> bfs(state);
    bfs.search(true, false);

    // DepthFirstSearch<QueensState> dfs(state);
    // dfs.search(true, false);
    
    std::cout << "time spend:" << time(nullptr) - t0 << std::endl;
    return 0;
}
