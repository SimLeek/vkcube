from vulkan import vkGetPhysicalDeviceFormatProperties, VK_IMAGE_TILING_LINEAR, VK_IMAGE_TILING_OPTIMAL, \
    VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, \
    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_VIEW_TYPE_2D, VkImageSubresourceRange, \
    VkImageViewCreateInfo, vkCreateImageView, VK_IMAGE_ASPECT_COLOR_BIT, VkAttachmentDescription, VK_SAMPLE_COUNT_1_BIT, \
    VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, \
    VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, \
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VkAttachmentReference, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, \
    VkSubpassDescription, VK_PIPELINE_BIND_POINT_GRAPHICS, VkSubpassDependency, VK_SUBPASS_EXTERNAL, \
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT, \
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VkRenderPassCreateInfo, vkCreateRenderPass


class VulkanRenderer(object):
    def __init__(self, vulkan_device, vulkan_swap):
        self.vulkan_device = vulkan_device
        self.vulkan_swap = vulkan_swap

        self.swap_chain_image_views = []
        self.render_pass = None

    def setup(self):
        self.__create_image_views()
        self.__create_render_pass()

    def __find_supported_format(self, candidates, tiling, feature):
        for i in candidates:
            props = vkGetPhysicalDeviceFormatProperties(self.vulkan_device.physical_device, i)

            if (tiling == VK_IMAGE_TILING_LINEAR and (
                    props.linearTilingFeatures & feature == feature
            )) or (tiling == VK_IMAGE_TILING_OPTIMAL and (
                    props.optimalTilingFeatures & feature == feature
            )):
                return i
        return -1

    @property
    def depth_format(self):
        return self.__find_supported_format(
            [
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D24_UNORM_S8_UINT,
            ],
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        )

    def create_image_view(self, image, im_format, aspect_flage, dimensions=VK_IMAGE_VIEW_TYPE_2D):
        ssr = VkImageSubresourceRange(
            aspectMask=aspect_flage,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )

        view_info = VkImageViewCreateInfo(
            image=image,
            viewType=dimensions,
            format=im_format,
            subresourceRange=ssr,
        )

        return vkCreateImageView(self.vulkan_device.logical_device, view_info, None)

    def __create_image_views(self):
        self.swap_chain_image_views = []

        for i, image in enumerate(self.vulkan_swap.swap_chain_images):
            self.swap_chain_image_views.append(
                self.create_image_view(
                    image, self.vulkan_swap.swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT
                )
            )

    def __create_render_pass(self):
        color_attachment = VkAttachmentDescription(
            format=self.vulkan_swap.swap_chain_image_format,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )

        depth_attachment = VkAttachmentDescription(
            format=self.depth_format,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

        color_attachment_ref = VkAttachmentReference(
            0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        depth_attachment_ref = VkAttachmentReference(
            1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        subpass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            pColorAttachments=[color_attachment_ref],
            pDepthStencilAttachment=[depth_attachment_ref],
        )

        dependency = VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                          | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )

        render_pass_info = VkRenderPassCreateInfo(
            pAttachments=[color_attachment, depth_attachment],
            pSubpasses=[subpass],
            pDependencies=[dependency],
        )

        self.render_pass = vkCreateRenderPass(
            self.vulkan_device.logical_device, render_pass_info, None
        )