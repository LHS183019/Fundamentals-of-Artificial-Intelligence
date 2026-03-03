@echo off
set start_time=%time%

g++ -o queens_simulated_anneal queens_simulated_anneal.cpp --std=c++11 -O3

for /l %%i in (200,1,210) do (
    echo Running with %%i queens...
    queens_simulated_anneal %%i >> output\queens_output_1.txt
)

exit /b