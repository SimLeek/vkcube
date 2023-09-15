from vulkan import (
    VkDebugReportCallbackCreateInfoEXT,
    VK_DEBUG_REPORT_WARNING_BIT_EXT,
    VK_DEBUG_REPORT_ERROR_BIT_EXT,
    VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
    VK_DEBUG_REPORT_DEBUG_BIT_EXT,
    VK_DEBUG_REPORT_INFORMATION_BIT_EXT
)

from kube.vkprochelp import vkCreateDebugReportCallbackEXT, vkDestroyDebugReportCallbackEXT


def debug_callback(*args):
    lvl = "?"
    if args[0] & VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
        lvl = "INFO"
    elif args[0] & VK_DEBUG_REPORT_WARNING_BIT_EXT:
        lvl = "WARNING"
    elif args[0] & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
        lvl = "PERFORMANCE"
    elif args[0] & VK_DEBUG_REPORT_ERROR_BIT_EXT:
        lvl = "ERROR"
    elif args[0] & VK_DEBUG_REPORT_DEBUG_BIT_EXT:
        lvl = "DEBUG"
    print(f"{lvl}: {args[5]} {args[6]}")
    return 0


class VulkanDebugger(object):

    def __init__(self, vulkan_instance, enable_validation_layers=True):
        self.vulkan_instance = vulkan_instance
        self.enable_validation_layers = enable_validation_layers

        self.__callback = None

    def setup(self):
        if not self.enable_validation_layers:
            return

        create_info = VkDebugReportCallbackCreateInfoEXT(
            flags=(
                    VK_DEBUG_REPORT_WARNING_BIT_EXT |
                    VK_DEBUG_REPORT_ERROR_BIT_EXT |
                    VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
                    VK_DEBUG_REPORT_DEBUG_BIT_EXT |
                    VK_DEBUG_REPORT_INFORMATION_BIT_EXT
            ),
            pfnCallback=debug_callback,
        )

        self.__callback = vkCreateDebugReportCallbackEXT(
            self.vulkan_instance.instance, create_info, None
        )

    def __del__(self):
        if self.__callback:
            vkDestroyDebugReportCallbackEXT(self.vulkan_instance.instance, self.__callback, None)