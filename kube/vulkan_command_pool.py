from vulkan import VkCommandPoolCreateInfo, vkCreateCommandPool

from kube.vulkan_device import VulkanDevice
from kube.vulkan_surface import VulkanSurface
from kube.vulkan_functions import find_queue_families


class VulkanCommandPool(object):
    def __init__(self, vulkan_device: VulkanDevice, vulkan_surface: VulkanSurface):
        self.vulkan_device = vulkan_device
        self.vulkan_surface = vulkan_surface
        self.pool = None

    def setup(self):
        self.__create_command_pool()

    def __create_command_pool(self):
        queue_family_indices = find_queue_families(self.vulkan_device.physical_device, self.vulkan_surface.surface)

        create_info = VkCommandPoolCreateInfo(
            queueFamilyIndex=queue_family_indices.graphics_family
        )

        self.pool = vkCreateCommandPool(
            self.vulkan_device.logical_device, create_info, None
        )