@echo off
setlocal

if "%ISAAC_SIM_ROOT%"=="" (
  echo ERROR: Set ISAAC_SIM_ROOT to your Isaac Sim installation folder.
  echo Example: set ISAAC_SIM_ROOT=C:\isaacsim
  exit /b 1
)

if not exist "%ISAAC_SIM_ROOT%\isaac-sim.bat" (
  echo ERROR: "%ISAAC_SIM_ROOT%\isaac-sim.bat" was not found.
  exit /b 1
)

if "%APG_ISAAC_CACHE%"=="" set "APG_ISAAC_CACHE=%LOCALAPPDATA%\APG-GS-Isaac\DerivedDataCache"
if not exist "%APG_ISAAC_CACHE%" mkdir "%APG_ISAAC_CACHE%"

echo Isaac cache: %APG_ISAAC_CACHE%
echo The launcher does not delete this folder automatically.
call "%ISAAC_SIM_ROOT%\isaac-sim.bat" --/UJITSO/datastore/localCachePath="%APG_ISAAC_CACHE%" %*
endlocal
