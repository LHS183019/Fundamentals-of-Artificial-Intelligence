@echo off

for /l %%i in (8,1,15) do (
    echo Running with %%i queens...
    python queens_bfs_dfs.py %%i >> queens_output.txt

    set current_time=%time%
    call :check_time %start_time% %current_time%
)

exit /b