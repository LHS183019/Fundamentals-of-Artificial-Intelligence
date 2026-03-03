@echo off

@REM for /l %%i in (1,1,5) do (
@REM     W400_QS4_solution 100000 >> W10_output.txt
@REM )
@REM for /l %%i in (1,1,5) do (
@REM     W400_QS4_solution 4000000 >> W400_output.txt
@REM )

for /l %%i in (1,1,5) do (
    W400_QS4_solution 10000000 >> W1000_output.txt
)
exit /b