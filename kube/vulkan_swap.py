from vulkan import VK_FORMAT_UNDEFINED, VK_FORMAT_B8G8R8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, \
    VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VkExtent2D, \
    VkSwapchainCreateInfoKHR, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_CONCURRENT, \
    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_SHARING_MODE_EXCLUSIVE

from kube.vkprochelp import vkCreateSwapchainKHR, vkGetSwapchainImagesKHR
from kube.vulkan_device import VulkanDevice
from kube.vulkan_functions import query_swap_chain_support, find_queue_families
from kube.vulkan_surface import VulkanSurface
from kube.vulkan_window import VulkanWindow


class VulkanSwap(object):
    def __init__(self, vulkan_device: VulkanDevice, vulkan_window:VulkanWindow, vulkan_surface:VulkanSurface):
        self.vulkan_device = vulkan_device
        self.vulkan_window = vulkan_window
        self.vulkan_surface = vulkan_surface

        self.swap_chain_images = []
        self.swap_chain_image_format = None
        self.swap_chain_extent = None

        self.swap_chain = None

    def setup(self):
        self.__create_swap_chain()

    def __choose_swap_surface_format(self, formats):
        if len(formats) == 1 and formats[0].format == VK_FORMAT_UNDEFINED:
            return [VK_FORMAT_B8G8R8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]

        for i in formats:
            if (
                    i.format == VK_FORMAT_B8G8R8_UNORM
                    and i.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                return i

        return formats[0]

    def __choose_swap_present_mode(self, present_modes):
        best_mode = VK_PRESENT_MODE_FIFO_KHR

        for i in present_modes:
            if (i == VK_PRESENT_MODE_FIFO_KHR) or (i == VK_PRESENT_MODE_MAILBOX_KHR) or (
                    i == VK_PRESENT_MODE_IMMEDIATE_KHR):
                return i

        return best_mode

    def __choose_swap_extent(self, capabilities):
        w, h = self.vulkan_window.get_window_size()
        width = max(
            capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, w)
        )
        height = max(
            capabilities.minImageExtent.height,
            min(capabilities.maxImageExtent.height, h),
        )
        return VkExtent2D(width, height)

    def __create_swap_chain(self):
        swap_chain_support = query_swap_chain_support(self.vulkan_device.physical_device, self.vulkan_surface.surface)

        surface_format = self.__choose_swap_surface_format(swap_chain_support.formats)
        present_mode = self.__choose_swap_present_mode(swap_chain_support.present_modes)
        extent = self.__choose_swap_extent(swap_chain_support.capabilities)

        image_count = swap_chain_support.capabilities.minImageCount + 1
        if (
                swap_chain_support.capabilities.maxImageCount > 0
                and image_count > swap_chain_support.capabilities.maxImageCount
        ):
            image_count = swap_chain_support.capabilities.maxImageCount

        indices = find_queue_families(self.vulkan_device.physical_device, self.vulkan_surface.surface)
        queue_family = {}.fromkeys([indices.graphics_family, indices.present_family])
        queue_families = list(queue_family.keys())
        if len(queue_families) > 1:
            create_info = VkSwapchainCreateInfoKHR(
                surface=self.vulkan_surface.surface,
                minImageCount=image_count,
                imageFormat=surface_format.format,
                imageColorSpace=surface_format.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                pQueueFamilyIndices=queue_families,
                imageSharingMode=VK_SHARING_MODE_CONCURRENT,
                preTransform=swap_chain_support.capabilities.currentTransform,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=present_mode,
                clipped=True,
            )
        else:
            create_info = VkSwapchainCreateInfoKHR(
                surface=self.vulkan_surface.surface,
                minImageCount=image_count,
                imageFormat=surface_format.format,
                imageColorSpace=surface_format.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                pQueueFamilyIndices=queue_families,
                imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
                preTransform=swap_chain_support.capabilities.currentTransform,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=present_mode,
                clipped=True,
            )

        self.swap_chain = vkCreateSwapchainKHR(self.vulkan_device.logical_device, create_info, None)
        assert self.swap_chain is not None

        self.swap_chain_images = vkGetSwapchainImagesKHR(
            self.vulkan_device.logical_device, self.swap_chain
        )

        self.swap_chain_image_format = surface_format.format
        self.swap_chain_extent = extent