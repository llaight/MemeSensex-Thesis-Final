@echo off
:loop
echo Starting InstaTunnel...
instatunnel connect 5001 --subdomain memesense-x --api-key it_6181dff6416ba1a417bc40f300e373569e5695c42fa0595d26a797079303
echo InstaTunnel exited. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop