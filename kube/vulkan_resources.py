from vulkan import vkDestroyImageView, vkDestroyImage, vkFreeMemory, VK_IMAGE_TILING_OPTIMAL, \
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, \
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VkFramebufferCreateInfo, \
    vkCreateFramebuffer


class VulkanResources(object):
    def __init__(self, parent):
        self.parent = parent

        self.__depth_image = None
        self.depth_image_memory = None
        self.__depth_image_view = None

        self.swap_chain_framebuffers = []

    def setup(self):
        self.__create_depth_resources()
        self.__create_frambuffers()

    def cleanup(self):
        vkDestroyImageView(self.parent.vulkan_device.logical_device, self.__depth_image_view, None)
        vkDestroyImage(self.parent.vulkan_device.logical_device, self.__depth_image, None)
        vkFreeMemory(self.parent.vulkan_device.logical_device, self.depth_image_memory, None)

    def __create_depth_resources(self):
        depth_format = self.parent.vulkan_renderer.depth_format

        self.__depth_image, self.depth_image_memory = self.parent.create_image(
            self.parent.vulkan_swap.swap_chain_extent.width,
            self.parent.vulkan_swap.swap_chain_extent.height,
            depth_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.__depth_image_view = self.parent.vulkan_renderer.create_image_view(
            self.__depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT
        )

        self.parent.transition_image_layout(
            self.__depth_image,
            depth_format,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            queue=self.parent.vulkan_device.graphic_queue
        )

    def __create_frambuffers(self):
        self.swap_chain_framebuffers = []
        for i, iv in enumerate(self.parent.vulkan_renderer.swap_chain_image_views):
            framebuffer_info = VkFramebufferCreateInfo(
                renderPass=self.parent.vulkan_renderer.render_pass,
                pAttachments=[iv, self.__depth_image_view],
                width=self.parent.vulkan_swap.swap_chain_extent.width,
                height=self.parent.vulkan_swap.swap_chain_extent.height,
                layers=1,
            )

            self.swap_chain_framebuffers.append(
                vkCreateFramebuffer(self.parent.vulkan_device.logical_device, framebuffer_info, None)
            )