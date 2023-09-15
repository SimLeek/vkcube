import sdl2
from vulkan import vkEnumerateInstanceExtensionProperties, VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VkApplicationInfo, \
    VK_STRUCTURE_TYPE_APPLICATION_INFO, VK_MAKE_VERSION, VK_API_VERSION, vkEnumerateInstanceLayerProperties, \
    VkInstanceCreateInfo, VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, vkCreateInstance

from kube.vkprochelp import InstanceProcAddr
from kube.vulkan_window import VulkanWindow


class VulkanInstance(object):
    """The Vulkan Instance"""

    def __init__(self, window: VulkanWindow, enable_validation_layers=True):
        self.window = window

        self.enable_validation_layers = enable_validation_layers
        self.validation_layers = []

        self.instance = None

    def setup(self):
        self._create_instance()

    def __get_required_extensions(self):
        extensions = [
            e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)
        ]

        if self.enable_validation_layers:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        if self.window.wm_info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
            extensions.append("VK_KHR_win32_surface")
        elif self.window.wm_info.subsystem == sdl2.SDL_SYSWM_X11:
            extensions.append("VK_KHR_xlib_surface")
        elif self.window.wm_info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
            extensions.append("VK_KHR_wayland_surface")
        else:
            raise SystemError("Platform not supported")

        return extensions

    def _create_instance(self):
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Python VKCube",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="pyvulkan",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION,
        )

        layers = vkEnumerateInstanceLayerProperties()
        layers = [l.layerName for l in layers]
        print("availables layers: %s\n" % layers)

        if self.enable_validation_layers:
            if "VK_LAYER_KHRONOS_validation" in layers:
                self.validation_layers = ["VK_LAYER_KHRONOS_validation"]
            elif "VK_LAYER_LUNARG_standard_validation" in layers:
                self.validation_layers = ["VK_LAYER_LUNARG_standard_validation"]
            else:
                raise SystemError("validation layers requested, but not available!")
        else:
            self.validation_layers = []

        extenstions = self.__get_required_extensions()
        if self.enable_validation_layers:
            instance_info = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                flags=0,
                pApplicationInfo=app_info,
                enabledLayerCount=len(self.validation_layers),
                ppEnabledLayerNames=self.validation_layers,
                enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions,
            )
        else:
            instance_info = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                flags=0,
                pApplicationInfo=app_info,
                enabledLayerCount=0,
                enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions,
            )

        self.instance = vkCreateInstance(instance_info, None)

        InstanceProcAddr.T = self.instance