from vulkan import VkPipelineVertexInputStateCreateInfo, VkPipelineInputAssemblyStateCreateInfo, \
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VkViewport, VkRect2D, VkPipelineViewportStateCreateInfo, \
    VkPipelineRasterizationStateCreateInfo, VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, \
    VkPipelineMultisampleStateCreateInfo, VK_SAMPLE_COUNT_1_BIT, VkPipelineDepthStencilStateCreateInfo, \
    VK_COMPARE_OP_LESS, VkPipelineColorBlendAttachmentState, VK_COLOR_COMPONENT_R_BIT, VK_COLOR_COMPONENT_G_BIT, \
    VK_COLOR_COMPONENT_B_BIT, VK_COLOR_COMPONENT_A_BIT, VkPipelineColorBlendStateCreateInfo, VK_LOGIC_OP_COPY, \
    VkPipelineLayoutCreateInfo, vkCreatePipelineLayout, VkGraphicsPipelineCreateInfo, VK_NULL_HANDLE, \
    vkCreateGraphicsPipelines, VkComputePipelineCreateInfo, vkCreateComputePipelines, vkDestroyPipelineLayout

from kube.vulkan_device import VulkanDevice
from kube.vulkan_renderer import VulkanRenderer
from kube.vulkan_swap import VulkanSwap


class VulkanPipelines(object):
    def __init__(self, vulkan_device:VulkanDevice, vulkan_swap:VulkanSwap, vulkan_renderer:VulkanRenderer):
        self.vulkan_device = vulkan_device
        self.vulkan_swap = vulkan_swap
        self.vulkan_renderer = vulkan_renderer
        self.pipeline = None
        self.pipeline_layout = None
        self.descriptor_set_layout = None
        self.shader_stage_infos = None
        self.compute_shader_stage_info = None

    def setup(self, shader_stage_infos, compute_shader_stage_info):
        self.shader_stage_infos = shader_stage_infos
        self.compute_shader_stage_info = compute_shader_stage_info

        self.__create_graphics_pipeline()

        self.__create_compute_pipeline()

    def reset_graphics(self):
        self.__create_graphics_pipeline()

    def __create_graphics_pipeline(self):
        vertex_input_info = VkPipelineVertexInputStateCreateInfo(
            pVertexBindingDescriptions=None,
            pVertexAttributeDescriptions=None,
        )

        input_assembly = VkPipelineInputAssemblyStateCreateInfo(
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, primitiveRestartEnable=False
        )

        viewport = VkViewport(
            0.0,
            0.0,
            float(self.vulkan_swap.swap_chain_extent.width),
            float(self.vulkan_swap.swap_chain_extent.height),
            0.0,
            1.0,
        )

        scissor = VkRect2D([0, 0], self.vulkan_swap.swap_chain_extent)
        viewport_stage = VkPipelineViewportStateCreateInfo(
            viewportCount=1, pViewports=viewport, scissorCount=1, pScissors=scissor
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=VK_CULL_MODE_BACK_BIT,
            frontFace=VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=False,
        )

        multisampling = VkPipelineMultisampleStateCreateInfo(
            sampleShadingEnable=False, rasterizationSamples=VK_SAMPLE_COUNT_1_BIT
        )

        depth_stencil = VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=True,
            depthWriteEnable=True,
            depthCompareOp=VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=False,
            stencilTestEnable=False,
        )

        color_blend_attachment = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT
                           | VK_COLOR_COMPONENT_G_BIT
                           | VK_COLOR_COMPONENT_B_BIT
                           | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=False,
        )

        color_bending = VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=color_blend_attachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0],
        )

        pipeline_layout_info = VkPipelineLayoutCreateInfo(
            pushConstantRangeCount=0,
            pSetLayouts=[self.descriptor_set_layout],
        )

        self.pipeline_layout = vkCreatePipelineLayout(
            self.vulkan_device.logical_device, pipeline_layout_info, None
        )

        if len(self.shader_stage_infos) == 0:
            raise RuntimeError("Please create shaders before running graphics")

        pipeline_info = VkGraphicsPipelineCreateInfo(
            pStages=self.shader_stage_infos,
            pVertexInputState=vertex_input_info,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_stage,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=color_bending,
            pDepthStencilState=depth_stencil,
            layout=self.pipeline_layout,
            renderPass=self.vulkan_renderer.render_pass,
            subpass=0,
            basePipelineHandle=VK_NULL_HANDLE,
        )

        self.pipeline = vkCreateGraphicsPipelines(
            self.vulkan_device.logical_device, VK_NULL_HANDLE, 1, pipeline_info, None
        )[0]

    def __create_compute_pipeline(self):
        pipeline_layout_info = VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )

        self.compute_pipeline_layout = vkCreatePipelineLayout(
            self.vulkan_device.logical_device, pipeline_layout_info, None
        )

        compute_pipeline_info = VkComputePipelineCreateInfo(
            stage=self.compute_shader_stage_info,
            layout=self.compute_pipeline_layout,
            basePipelineHandle=VK_NULL_HANDLE,
            basePipelineIndex=-1,
        )

        self.compute_pipeline = vkCreateComputePipelines(
            self.vulkan_device.logical_device, VK_NULL_HANDLE, 1, compute_pipeline_info, None
        )[0]

        # Destroy the graphics pipeline layout if it's no longer needed
        if self.compute_pipeline_layout:
            vkDestroyPipelineLayout(self.vulkan_device.logical_device, self.compute_pipeline_layout, None)

        return self.compute_pipeline