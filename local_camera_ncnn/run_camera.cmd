@echo off
"%~dp0dist\local_camera.exe" %*
if errorlevel 1 pause
