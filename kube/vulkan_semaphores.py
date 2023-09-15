from vulkan import VkSemaphoreCreateInfo, vkCreateSemaphore, vkCreateFence, VK_FENCE_CREATE_SIGNALED_BIT

from kube.vulkan_device import VulkanDevice


class VulkanSemaphores(object):
    def __init__(self, vulkan_device: VulkanDevice):
        self.vulkan_device = vulkan_device
        self.image_available_semaphore = None
        self.render_finished_semaphore = None
        self.MAX_FRAMES_IN_FLIGHT = 1

    def setup(self):
        self.__create_semaphores()

    def __create_semaphores(self):
        semaphore_info = VkSemaphoreCreateInfo()

        self.image_available_semaphore = vkCreateSemaphore(
            self.vulkan_device.logical_device, semaphore_info, None
        )
        self.render_finished_semaphore = vkCreateSemaphore(
            self.vulkan_device.logical_device, semaphore_info, None
        )
        self.write_semaphore = vkCreateSemaphore(self.vulkan_device.logical_device, semaphore_info, None)
        self.read_semaphore = vkCreateSemaphore(self.vulkan_device.logical_device, semaphore_info, None)
        self.in_flight_fences = [vkCreateFence(self.vulkan_device.logical_device, VK_FENCE_CREATE_SIGNALED_BIT, None)
                                 for _ in
                                 range(self.MAX_FRAMES_IN_FLIGHT)]