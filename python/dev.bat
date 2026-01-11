@echo off
REM Samurai Python Bindings - Development Helper Script (Windows)
REM
REM This script provides convenient commands for developing the Python bindings.
REM
REM Usage:
REM   dev.bat build              # Build the module
REM   dev.bat install            # Install in development mode
REM   dev.bat test               # Run tests
REM   dev.bat clean              # Clean build artifacts
REM   dev.bat reinstall          # Clean and reinstall
REM   dev.bat all                # Build + install + test

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set BUILD_TYPE=Release

REM Parse arguments
if "%1"=="debug" set BUILD_TYPE=Debug
if "%2"=="debug" set BUILD_TYPE=Debug

if "%1"=="build" goto :build
if "%1"=="install" goto :install
if "%1"=="test" goto :test
if "%1"=="clean" goto :clean
if "%1"=="reinstall" goto :reinstall
if "%1"=="all" goto :all

REM Show usage if no command specified
echo.
echo Samurai Python Bindings - Development Helper
echo.
echo Usage:
echo   dev.bat build              # Build the module
echo   dev.bat install            # Install in development mode
echo   dev.bat test               # Run tests
echo   dev.bat clean              # Clean build artifacts
echo   dev.bat reinstall          # Clean and reinstall
echo   dev.bat all                # Build + install + test
echo   dev.bat [command] debug    # Use Debug configuration
echo.
goto :end

:build
echo.
echo ========================================
echo Building Python Module
echo ========================================
echo.

if not exist build mkdir build
echo Configuring CMake...
cmake -B build -S . -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DSAMURAI_PYTHON_STANDALONE=ON -A x64
if errorlevel 1 goto :error

echo.
echo Building...
cmake --build build --config %BUILD_TYPE%
if errorlevel 1 goto :error

echo.
echo ========================================
echo Build Complete
echo ========================================
goto :end

:install
echo.
echo ========================================
echo Installing Python Module
echo ========================================
echo.

python -m pip install -e .
if errorlevel 1 goto :error

echo.
echo ========================================
echo Installation Complete
echo ========================================
goto :end

:test
echo.
echo ========================================
echo Running Tests
echo ========================================
echo.

python -m pytest tests -v
if errorlevel 1 goto :error

echo.
echo ========================================
echo Tests Complete
echo ========================================
goto :end

:clean
echo.
echo ========================================
echo Cleaning Build Artifacts
echo ========================================
echo.

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%d in (*.egg-info) do rmdir /s /q "%%d"
if exist __pycache__ rmdir /s /q __pycache__
if exist src\__pycache__ rmdir /s /q src\__pycache__
if exist src\samurai_python\__pycache__ rmdir /s /q src\samurai_python\__pycache__
if exist tests\__pycache__ rmdir /s /q tests\__pycache__

echo.
echo ========================================
echo Clean Complete
echo ========================================
goto :end

:reinstall
echo.
echo ========================================
echo Reinstalling Python Module
echo ========================================
echo.

call :clean
call :build
call :install
goto :end

:all
echo.
echo ========================================
echo Build + Install + Test
echo ========================================
echo.

call :build
call :install
call :test
goto :end

:error
echo.
echo ========================================
echo ERROR: Command failed
echo ========================================
exit /b 1

:end
endlocal
