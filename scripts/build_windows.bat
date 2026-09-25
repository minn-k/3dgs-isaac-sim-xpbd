@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "BUILD_DIR=%REPO_ROOT%\build\native"
if "%CUDA_ARCHS%"=="" set "CUDA_ARCHS=86"
if "%CMAKE_GENERATOR%"=="" set "CMAKE_GENERATOR=Visual Studio 17 2022"

cmake -S "%REPO_ROOT%\native" -B "%BUILD_DIR%" -G "%CMAKE_GENERATOR%" -A x64 -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCHS%
if errorlevel 1 exit /b %errorlevel%

cmake --build "%BUILD_DIR%" --config Release --target xpbd_isaac
exit /b %errorlevel%
