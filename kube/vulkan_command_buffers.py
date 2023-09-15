from vulkan import VkCommandBufferAllocateInfo, VK_COMMAND_BUFFER_LEVEL_PRIMARY, vkAllocateCommandBuffers, \
    VkCommandBufferBeginInfo, VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT, vkBeginCommandBuffer, vkCmdBindPipeline, \
    VK_PIPELINE_BIND_POINT_COMPUTE, vkCmdDispatch, VkMemoryBarrier, VK_ACCESS_SHADER_WRITE_BIT, \
    VK_ACCESS_SHADER_READ_BIT, vkCmdPipelineBarrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, \
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, vkEndCommandBuffer, VkRect2D, VkClearValue, \
    VkRenderPassBeginInfo, vkCmdBeginRenderPass, VK_SUBPASS_CONTENTS_INLINE, VK_PIPELINE_BIND_POINT_GRAPHICS, \
    vkCmdBindDescriptorSets, vkCmdDraw, vkCmdEndRenderPass

from kube.vulkan_command_pool import VulkanCommandPool
from kube.vulkan_descriptors import VulkanDescriptors
from kube.vulkan_device import VulkanDevice
from kube.vulkan_pipelines import VulkanPipelines
from kube.vulkan_renderer import VulkanRenderer
from kube.vulkan_resources import VulkanResources
from kube.vulkan_swap import VulkanSwap


class VulkanCommandBuffers(object):
    def __init__(self,
                 vulkan_device: VulkanDevice,
                 vulkan_command_pool: VulkanCommandPool,
                 vulkan_pipelines: VulkanPipelines,
                 vulkan_resources: VulkanResources,
                 vulkan_swap: VulkanSwap,
                 vulkan_renderer: VulkanRenderer,
                 vulkan_descriptors: VulkanDescriptors,
                 compute_groups):
        self.vulkan_device = vulkan_device
        self.vulkan_command_pool = vulkan_command_pool
        self.vulkan_pipelines = vulkan_pipelines
        self.vulkan_resources = vulkan_resources
        self.vulkan_swap = vulkan_swap
        self.vulkan_renderer = vulkan_renderer
        self.vulkan_descriptors = vulkan_descriptors

        self.compute_group_counts = compute_groups

        self.num_compute_command_buffers = 1

    def setup(self):
        self.__create_compute_command_buffers()
        self.__create_command_buffers()

    def reset_graphics(self):
        self.__create_command_buffers()

    def __create_compute_command_buffers(self):
        self.__compute_command_buffers = []

        command_buffer_allocate_info = VkCommandBufferAllocateInfo(
            commandPool=self.vulkan_command_pool.pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,  # Use PRIMARY for one-time use
            commandBufferCount=self.num_compute_command_buffers,  # Specify the number of command buffers you need
        )

        self.__compute_command_buffers = vkAllocateCommandBuffers(
            self.vulkan_device.logical_device, command_buffer_allocate_info
        )

        for i, compute_command_buffer in enumerate(self.__compute_command_buffers):
            # Begin recording the command buffer
            command_buffer_begin_info = VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                pInheritanceInfo=None,  # Set to None for primary command buffers
            )

            vkBeginCommandBuffer(compute_command_buffer, command_buffer_begin_info)

            # Bind the compute pipeline
            vkCmdBindPipeline(
                compute_command_buffer,
                pipelineBindPoint=VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline=self.vulkan_pipelines.compute_pipeline,
            )

            # Bind descriptor sets if needed
            # vkCmdBindDescriptorSets(compute_command_buffer, ...)

            # Dispatch the compute work (specify the number of workgroups)
            vkCmdDispatch(compute_command_buffer, *self.compute_group_counts)

            self.compute_to_graphics_barrier = VkMemoryBarrier(
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,  # Compute shader writes
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,  # Graphics shader reads
            )

            vkCmdPipelineBarrier(
                compute_command_buffer,
                srcStageMask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                dependencyFlags=0,
                memoryBarrierCount=1,
                pMemoryBarriers=[self.compute_to_graphics_barrier],
                bufferMemoryBarrierCount=0,
                pBufferMemoryBarriers=[],  # You might need this for buffer transitions
                imageMemoryBarrierCount=0,
                pImageMemoryBarriers=[],  # You might need this for image transitions
            )

            # End recording the command buffer
            vkEndCommandBuffer(compute_command_buffer)

    def __create_command_buffers(self):
        self.__command_buffers = []

        alloc_info = VkCommandBufferAllocateInfo(
            commandPool=self.vulkan_command_pool.pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.vulkan_resources.swap_chain_framebuffers),
        )

        self.__command_buffers = vkAllocateCommandBuffers(
            self.vulkan_device.logical_device, alloc_info
        )

        for i, buffer in enumerate(self.__command_buffers):
            begin_info = VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
            )
            vkBeginCommandBuffer(buffer, begin_info)

            render_area = VkRect2D([0, 0], self.vulkan_swap.swap_chain_extent)
            clear_color = [
                VkClearValue(color=[[0.0, 0.0, 0.0, 1.0]]),
                VkClearValue(depthStencil=[1.0, 0]),
            ]
            render_pass_info = VkRenderPassBeginInfo(
                renderPass=self.vulkan_renderer.render_pass,
                framebuffer=self.vulkan_resources.swap_chain_framebuffers[i],
                renderArea=render_area,
                pClearValues=clear_color,
            )

            vkCmdBeginRenderPass(buffer, render_pass_info, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.vulkan_pipelines.pipeline)

            vkCmdBindDescriptorSets(
                buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.vulkan_pipelines.pipeline_layout,
                0,
                1,
                self.vulkan_descriptors.descriptor_set,
                0,
                None,
            )

            vkCmdDraw(buffer, 3, 1, 0, 0)

            vkCmdPipelineBarrier(
                buffer,
                srcStageMask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  # Source stage from compute
                dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,  # Destination stage for fragment
                dependencyFlags=0,
                pMemoryBarriers=[self.compute_to_graphics_barrier],  # If you have memory barriers
                memoryBarrierCount=1,
                pBufferMemoryBarriers=[],  # If you have buffer memory barriers
                bufferMemoryBarrierCount=0,
                pImageMemoryBarriers=[],  # If you have image memory barriers
                imageMemoryBarrierCount=0
            )

            vkCmdEndRenderPass(buffer)

            vkEndCommandBuffer(buffer)