@echo off
for %%i in (*.vert *.frag *.comp *.geom *.tese *.tesc) do "C:\VulkanSDK\1.3.250.1\Bin\glslc.exe" "%%~i" -o "%%~i.spv"