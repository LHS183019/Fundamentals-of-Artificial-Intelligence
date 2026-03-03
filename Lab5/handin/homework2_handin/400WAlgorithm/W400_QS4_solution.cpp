/*
QS4算法是由Rok Sosic和Jun Gu于1994年提出的用于解决N皇后问题的算法, 属于局部搜索算法，时间复杂度可以达到O(N)，且不像构造法对皇后奇偶数有额外要求。
其核心是分开“初始化+迭代修复”两个阶段，初始化阶段利用了最小冲突启发式（Min-Conflict Heuristic）策略，修复阶段则是贪心+局部搜索的思维。
Sosic和Gu原始论文：https://www.dcc.fc.up.pt/~ines/aulas/1415/IA/Papers/p22-sosic.pdf

我的代码是改写自以下repo的，进行了针对c++的一些优化。
Python实现参考：https://github.com/duncansclarke/N-Queens

我在W400_QS4_solution.md里写了更多的算法心得和性能评测.
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

// 全局变量, 代表皇后的数量
int SIZE = 0;

std::vector<int> pieces; //存储每个皇后的行位置(列位置固定)
std::vector<int> x_conflicts; //用以检测行冲突，储存某行的皇后数量
std::vector<int> y_conflicts; //用以检测列冲突，储存某列的皇后数量
std::vector<int> neg_conflicts; //用以检测负对角线冲突，同一条负对角线上的所有位置，行-列的值相同，再加上偏移量SIZE-1转换为正数索引
std::vector<int> pos_conflicts; //用以检测正对角线冲突，同一条正对角线上的所有位置，行+列的值相同

std::mt19937 gen; //用来生成随机数，注意:如果用srand()是不行的，我们需要高质量的随机数

// 交换皇后i和皇后j的行位置
inline void swap(int i, int j) {
    int storage = pieces[i];
    pieces[i] = pieces[j];
    pieces[j] = storage;
}

// 计算皇后i所在对角线冲突数
inline int partial_collisions(int i) {
    int neg = neg_conflicts[pieces[i] - i + SIZE -1];
    int pos = pos_conflicts[pieces[i] + i];
    return neg + pos;
}

// 皇后i所在行、列、对角线，自增冲突计数
void add_conflict(int i) {
    neg_conflicts[pieces[i] - i + SIZE -1]++;
    pos_conflicts[pieces[i] + i]++;
    x_conflicts[i]++;
    y_conflicts[pieces[i]]++;
}

// 皇后i所在行、列、对角线，自减冲突计数
void remove_conflict(int i) {
    neg_conflicts[pieces[i] - i + SIZE -1]--;
    pos_conflicts[pieces[i] + i]--;
    x_conflicts[i]--;
    y_conflicts[pieces[i]]--;
}

// 计算皇后i所在行、列、对角线总冲突数
inline int total_collisions(int i) {
    int x = x_conflicts[i];
    int y = y_conflicts[pieces[i]];
    int neg = neg_conflicts[pieces[i] - i + SIZE -1];
    int pos = pos_conflicts[pieces[i] + i];
    return x + y + neg + pos;
}

// 带冲突处理的交换
void swap_with_conflict(int i, int j) {
    remove_conflict(i);
    remove_conflict(j);
    swap(i, j);
    add_conflict(i);
    add_conflict(j);
}

// QS4算法的分层初始化(分两阶段 每阶段放置策略不同)
int initial_search() {
    // 初始化，所有皇后分别放置不同行/列
    pieces.clear();
    for (int i = 0; i < SIZE; i++) {
        pieces.push_back(i);
    }

    /*
    最小冲突思想，从j后面随机选择一个行放j皇后，
    检查是否产生冲突(partial_collisions(j) - 2 == 0)
    如果无冲突，确认放置并更新冲突计数
    如果有冲突，撤销操作并继续尝试
    最多尝试3.08 × SIZE次(论文里面给出的超参数)
    总之，就是很贪婪地把一只皇后放到不会引起冲突的地方
    */
    int j = 0; // 用来记录第一个(左到右列数最前面的)有冲突的皇后的位置
    for (int i = 0; i < static_cast<int>(floor(3.08 * SIZE)); i++) {
        std::uniform_int_distribution<int> dis(j, SIZE-1);
        int m = dis(gen);
        swap(m, j);
        
        // 刚初始化完，第一步swap所有的皇后都不会行或列冲突
        neg_conflicts[pieces[j] - j + SIZE - 1]++;
        pos_conflicts[pieces[j] + j]++;
        
        if (partial_collisions(j) - 2 == 0) {
            x_conflicts[j]++;
            y_conflicts[pieces[j]]++;
            j++;
        } else {
            neg_conflicts[pieces[j] - j + SIZE - 1]--;
            pos_conflicts[pieces[j] + j]--;
            swap(m, j);
        }
        // 如果放好所有皇后且刚好没有冲突
        if (j == SIZE) {
            break;
        }
    }
    
    /*
    超出尝试次数后，剩下没处理的皇后直接随机放置
    */
    for (int i = j; i < SIZE; i++) {
        std::uniform_int_distribution<int> dis2(i, SIZE-1);
        int m = dis2(gen);
        swap(m, i);
        add_conflict(i);
    }
    
    return SIZE - j; // 剩下需要被处理的皇后数
}

/*
迭代修复阶段
逐个处理存在冲突的皇后
尝试将其与棋盘上的任意其他皇后进行交换；
关键是，只有当交换后两个皇后都完全没有冲突，才会接受这个交换，不同于一般的约束最小化；
完成交换后，就接着处理下一个存在冲突的皇后；
如果在给定的步数（2.0 * SIZE / log2(SIZE)）限制内无法完成求解（交换次数过多），
则停止求解，并返回空列表，当前棋盘已经陷入了局部最优解。
*/
std::vector<int> final_search(int k, int steps) {
    int s = 0;
    for (int i = SIZE - k; i < SIZE; i++) {
        if (total_collisions(i) - 4 > 0) {
            bool b = true; // 记录冲突是否存在
            while (b) {
                if (s == steps) {
                    return std::vector<int>();
                }
                std::uniform_int_distribution<int> dis(0, SIZE-1);
                int j = dis(gen);
                swap_with_conflict(i, j);
                
                // 如果仍存在冲突，取消交换操作
                b = (total_collisions(i) - 4 > 0) || (total_collisions(j) - 4 > 0);
                if (b) {
                    swap_with_conflict(i, j);
                }
                s++;
            }
        }
    }
    return pieces;
}

// 重置所有全局变量，这里用assign, reserve，为的是避免push_back操作带来的额外开销
void init_all() {
    x_conflicts.assign(SIZE, 0);
    y_conflicts.assign(SIZE, 0);
    neg_conflicts.assign(2 * SIZE - 1, 0);
    pos_conflicts.assign(2 * SIZE - 1, 0);
    pieces.clear();
    pieces.reserve(SIZE);
}

// QS4算法
std::vector<int> solver(int size) {
    SIZE = size;
    std::vector<int> solution;
    while (solution.empty()) { //while找不到解/被检查为局部最优解
        init_all();
        solution = final_search(initial_search(), static_cast<int>(ceil(2.0 * SIZE / log2(SIZE))));
    }
    return solution;
}

// 生成解的字符串表达，用于输出
std::string generate_output_string(const std::vector<int>& solution) {
    std::string output_string = "[" + std::to_string(solution[0] + 1);
    for (size_t i = 1; i < solution.size(); i++) {
        output_string += "," + std::to_string(solution[i] + 1);
    }
    output_string += "]";
    return output_string;
}

// 用来debug，确保算法正确性
void solution_checker(const std::vector<int>& solution) {
    std::vector<int> x_conf, y_conf, neg_conf, pos_conf;
    x_conf.assign(SIZE, 0);
    y_conf.assign(SIZE, 0);
    neg_conf.assign(2 * SIZE - 1, 0);
    pos_conf.assign(2 * SIZE - 1, 0);
    for (size_t x = 0; x < solution.size(); x++) {
        x_conf[x]++;
        y_conf[solution[x]]++;
        neg_conf[solution[x] - x + SIZE - 1]++;
        pos_conf[solution[x] + x]++;
    }
    
    for (size_t x = 0; x < solution.size(); x++) {
        int conflicts = x_conf[x] + y_conf[solution[x]] + neg_conf[solution[x] - x + SIZE - 1] + pos_conf[solution[x] + x];
        if (conflicts != 4) {
            std::cout << "failed...." << x << std::endl;
            return;
        }
    }
    std::cout << "test pass!" << std::endl;
}

int main(int argc, char* argv[]){
    
    std::random_device rd;
    gen.seed(rd());

    // 处理命令行参数
    int queens_cnt = 1000;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <queens_cnt>" << std::endl;
        return 1;
    }
    
    try {
        queens_cnt = std::stoi(argv[1]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << argv[1] << " is not a valid integer." << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << argv[1] << " is out of range." << std::endl;
        return 1;
    }

    // 计时器
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> solution = solver(queens_cnt);
    
    // 输出解
    std::ofstream output_file("nqueens_out.txt");
    output_file << generate_output_string(solution) << std::endl;
    output_file.close();

    // 打印耗时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "spend time: " << duration.count() << std::endl;

    // 检查解的正确性
    solution_checker(solution);
    
    return 0;
}
