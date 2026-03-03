// 状态转移:
    new_state = state.next(action);
// 更新状态价值，扩展待访问的节点集:
    best_value_of[new_state] = value_of(new_state);
    states_queue.push(new_state);
// 记录路径:
    last_state_of[new_state] = state;
