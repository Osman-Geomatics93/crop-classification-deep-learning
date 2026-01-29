@echo off
echo ============================================================
echo  Registering geodl conda environment as Jupyter kernel
echo ============================================================
echo.
echo This fixes the fbgemm.dll error by ensuring Jupyter uses
echo the correct conda environment with PyTorch installed.
echo.

C:\Users\wwwos\anaconda3\envs\geodl\python.exe -m ipykernel install --user --name geodl --display-name "Python (geodl)"
if errorlevel 1 (
    echo ERROR: ipykernel installation failed.
    echo Try: conda install -n geodl ipykernel
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  SUCCESS! Kernel "Python (geodl)" is now available.
echo  In Jupyter, select Kernel -> Change Kernel -> Python (geodl)
echo ============================================================
pause
