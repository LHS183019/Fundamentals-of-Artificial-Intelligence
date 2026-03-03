@echo off
set start_time=%time%

g++ -o queens_bfs_dfs queens_bfs_dfs.cpp --std=c++11 -O3

for /l %%i in (8,1,15) do (
    echo Running with %%i queens...
    queens_bfs_dfs %%i >> queens_output.txt
)

exit /b