from vulkan import VK_KHR_SWAPCHAIN_EXTENSION_NAME, vkEnumerateDeviceExtensionProperties, vkGetPhysicalDeviceFeatures, \
    vkEnumeratePhysicalDevices, VkDeviceQueueCreateInfo, VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, \
    VkPhysicalDeviceFeatures, VkDeviceCreateInfo, vkCreateDevice, vkGetDeviceQueue

from kube.vkprochelp import DeviceProcAddr
from kube.vulkan_functions import find_queue_families, query_swap_chain_support


class VulkanDevice(object):
    def __init__(self, vulkan_instance, vulkan_surface, enable_validation_layers=True):
        self.vulkan_instance = vulkan_instance
        self.vulkan_surface = vulkan_surface
        self.device_extensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        self.physical_device = None
        self.logical_device = None
        self.enable_validation_layers = enable_validation_layers

        self.graphic_queue = None
        self.present_queue = None
        self.compute_queue = None

    def setup(self):
        self._pick_physical_device()
        self.__create_logical_device()

    def __check_device_extension_support(self, device):
        available_extensions = vkEnumerateDeviceExtensionProperties(device, None)

        aen = [i.extensionName for i in available_extensions]
        for i in self.device_extensions:
            if i not in aen:
                return False

        return True

    def __is_device_suitable(self, device):
        indices = find_queue_families(device, self.vulkan_surface.surface)

        extensions_supported = self.__check_device_extension_support(device)

        swap_chain_adequate = False
        if extensions_supported:
            swap_chain_support = query_swap_chain_support(device, self.vulkan_surface.surface)
            swap_chain_adequate = (swap_chain_support.formats is not None) and (
                    swap_chain_support.present_modes is not None
            )

        supported_features = vkGetPhysicalDeviceFeatures(device)

        return (
                indices.is_complete
                and extensions_supported
                and swap_chain_adequate
                and supported_features.samplerAnisotropy
        )

    def _pick_physical_device(self):
        physical_devices = vkEnumeratePhysicalDevices(self.vulkan_instance.instance)

        for e, device in enumerate(physical_devices):
            if self.__is_device_suitable(device):
                self.physical_device = device
                self.device_num = e  # use this to ensure other GPU, if any, is used for compute shaders
                break

        assert self.physical_device is not None

    def __create_logical_device(self):
        indices = find_queue_families(self.physical_device, self.vulkan_surface.surface)

        unique_queue_families = {}.fromkeys(
            [indices.graphics_family, indices.present_family]
        )
        queue_create_infos = []
        for i in unique_queue_families:
            queue_create_info = VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=i,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            queue_create_infos.append(queue_create_info)

        device_features = VkPhysicalDeviceFeatures()
        device_features.samplerAnisotropy = True
        if self.enable_validation_layers:
            create_info = VkDeviceCreateInfo(
                queueCreateInfoCount=len(queue_create_infos),
                pQueueCreateInfos=queue_create_infos,
                enabledExtensionCount=len(self.device_extensions),
                ppEnabledExtensionNames=self.device_extensions,
                enabledLayerCount=len(self.vulkan_instance.validation_layers),
                ppEnabledLayerNames=self.vulkan_instance.validation_layers,
                pEnabledFeatures=device_features,
            )
        else:
            create_info = VkDeviceCreateInfo(
                queueCreateInfoCount=len(queue_create_infos),
                pQueueCreateInfos=queue_create_infos,
                enabledExtensionCount=len(self.device_extensions),
                ppEnabledExtensionNames=self.device_extensions,
                enabledLayerCount=0,
                pEnabledFeatures=device_features,
            )

        self.logical_device = vkCreateDevice(self.physical_device, create_info, None)

        DeviceProcAddr.T = self.logical_device

        self.graphic_queue = vkGetDeviceQueue(
            self.logical_device, indices.graphics_family, 0
        )
        self.present_queue = vkGetDeviceQueue(
            self.logical_device, indices.present_family, 0
        )
        self.compute_queue = vkGetDeviceQueue(
            self.logical_device, indices.compute_family, 0
        )