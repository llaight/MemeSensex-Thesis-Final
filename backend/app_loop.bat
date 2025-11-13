@echo off
:loop
python app.py
timeout /t 5 /nobreak >nul
goto loop