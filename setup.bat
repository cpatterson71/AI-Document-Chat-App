@echo off
cd /d "%~dp0"

:: This script automates the setup of the Haystack AI Document Query project on a new system.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and add it to your PATH.
    goto :eof
)

:: Check for Python 3.11
python --version 2>&1 | findstr "3.11" >nul
if %errorlevel% neq 0 (
    echo This script requires Python 3.11. Please install Python 3.11 and make sure it's in your PATH.
    python --version
    pause
    exit /b
)

:: Create a virtual environment
echo Creating a virtual environment...
python -m venv .venv

:: Activate the virtual environment and install dependencies
echo Installing dependencies from requirements.txt...
call .venv\Scripts\activate.bat
pip install -r requirements.txt

:: Check if .env file exists
if exist .env (
    echo .env file already exists. Skipping API key prompt.
) else (
    :: Prompt for the OpenAI API Key
    echo.
    echo Please enter your OpenAI API Key:
    set /p OPENAI_API_KEY=

    :: Create the .env file
    echo OPENAI_API_KEY=%OPENAI_API_KEY% > .env
)

echo.
echo Setup complete!
echo You can now run the application by executing the run_qa.bat file.
pause