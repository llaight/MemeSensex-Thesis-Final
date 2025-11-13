@echo off
:watch
echo Launching InstaTunnel loop...
cmd /c instatunnel-loop.bat
echo Main batch exited. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto watch