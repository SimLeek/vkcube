import os
import platform

arc = platform.architecture()
dir_path = None
if "Windows" in arc[1]:
    if arc[0] == "64bit":
        dir_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "bin64"
    elif arc[0] == "32bit":
        dir_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "bin32"
if dir_path is not None:
    os.environ["PYSDL2_DLL_PATH"] = dir_path
