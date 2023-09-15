import abc
import time

import numpy as np
from PIL import Image
from vulkan import (
    vkDeviceWaitIdle,
    vkDestroySampler,
    vkDestroyImageView,
    vkDestroyImage,
    vkFreeMemory,
    vkDestroyDescriptorPool,
    vkDestroyBuffer,
    vkDestroySemaphore,
    vkDestroyDescriptorSetLayout,
    vkDestroyCommandPool,
    vkDestroyDevice,
    vkDestroyInstance,
    vkDestroyFramebuffer,
    vkFreeCommandBuffers,
    vkDestroyPipeline,
    vkDestroyPipelineLayout,
    vkDestroyRenderPass,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_IMAGE_ASPECT_COLOR_BIT,
    VK_SAMPLE_COUNT_1_BIT,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    VkPipelineShaderStageCreateInfo,
    VK_SHADER_STAGE_VERTEX_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VK_NULL_HANDLE,
    vkDestroyShaderModule,
    VK_IMAGE_TILING_OPTIMAL,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    vkMapMemory,
    ffi,
    vkUnmapMemory,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VkSamplerCreateInfo,
    VK_FILTER_LINEAR,
    VK_SAMPLER_ADDRESS_MODE_REPEAT,
    VK_COMPARE_OP_ALWAYS,
    VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    vkCreateSampler,
    VkImageSubresourceRange,
    VkImageCreateInfo,
    VK_IMAGE_TYPE_2D,
    vkCreateImage,
    vkGetImageMemoryRequirements,
    VkMemoryAllocateInfo,
    vkAllocateMemory,
    vkBindImageMemory,
    VK_IMAGE_ASPECT_STENCIL_BIT,
    VkImageMemoryBarrier,
    VK_QUEUE_FAMILY_IGNORED,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_ACCESS_SHADER_READ_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
    vkCmdPipelineBarrier,
    VkImageSubresourceLayers,
    VkBufferImageCopy,
    vkCmdCopyBufferToImage,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VkDescriptorPoolSize,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VkDescriptorBufferInfo,
    VkDescriptorImageInfo,
    VkWriteDescriptorSet,
    VkBufferCreateInfo,
    vkCreateBuffer,
    vkGetBufferMemoryRequirements,
    vkBindBufferMemory,
    VkCommandBufferAllocateInfo,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    vkAllocateCommandBuffers,
    VkCommandBufferBeginInfo,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    vkBeginCommandBuffer,
    vkEndCommandBuffer,
    VkSubmitInfo,
    vkQueueSubmit,
    vkQueueWaitIdle,
    VkBufferCopy,
    vkCmdCopyBuffer,
    vkGetPhysicalDeviceMemoryProperties,
    VkErrorSurfaceLostKhr,
    VkPresentInfoKHR,
    VkErrorOutOfDateKhr,
    VkShaderModuleCreateInfo,
    vkCreateShaderModule,
    VK_IMAGE_TYPE_3D, VK_FORMAT_R8_SINT, VK_IMAGE_VIEW_TYPE_3D, VK_FORMAT_R8_UINT, VK_FORMAT_R8_UNORM
)

from .vkprochelp import (vkDestroySwapchainKHR,
                         vkAcquireNextImageKHR, vkQueuePresentKHR)
from .vulkan_command_buffers import VulkanCommandBuffers
from .vulkan_command_pool import VulkanCommandPool
from .vulkan_debugger import VulkanDebugger
from .vulkan_descriptors import VulkanDescriptors
from .vulkan_device import VulkanDevice
from .vulkan_instance import VulkanInstance
from .vulkan_pipelines import VulkanPipelines
from .vulkan_renderer import VulkanRenderer
from .vulkan_resources import VulkanResources
from .vulkan_semaphores import VulkanSemaphores
from .vulkan_surface import VulkanSurface
from .vulkan_swap import VulkanSwap
from .vulkan_window import VulkanWindow


class VKAbstractComputer:
    def __init__(self, width=1280, height=720, compute_groups=(10, 10, 10), enable_validation_layers=True):
        super(VKAbstractComputer, self).__init__()

        self.enable_validation_layers = enable_validation_layers
        self.vulkan_window = VulkanWindow(self, width, height)
        self.vulkan_instance = VulkanInstance(self.vulkan_window, self.enable_validation_layers)
        self.vulkan_debugger = VulkanDebugger(self.vulkan_instance, self.enable_validation_layers)
        self.vulkan_surface = VulkanSurface(self.vulkan_instance, self.vulkan_window)
        self.vulkan_device = VulkanDevice(self.vulkan_instance, self.vulkan_surface, self.enable_validation_layers)
        self.vulkan_swap = VulkanSwap(self.vulkan_device, self.vulkan_window, self.vulkan_surface)
        self.vulkan_renderer = VulkanRenderer(self.vulkan_device, self.vulkan_swap)
        self.vulkan_pipelines = VulkanPipelines(self.vulkan_device, self.vulkan_swap, self.vulkan_renderer)
        self.vulkan_command_pool = VulkanCommandPool(self.vulkan_device, self.vulkan_surface)
        self.vulkan_resources = VulkanResources(self)
        self.vulkan_descriptors = VulkanDescriptors(self.vulkan_device, self.vulkan_pipelines)
        self.vulkan_command_buffers = VulkanCommandBuffers(self.vulkan_device,
                                                           self.vulkan_command_pool,
                                                           self.vulkan_pipelines,
                                                           self.vulkan_resources,
                                                           self.vulkan_swap,
                                                           self.vulkan_renderer,
                                                           self.vulkan_descriptors,
                                                           compute_groups)
        self.vulkan_semaphores = VulkanSemaphores(self.vulkan_device)

        self.current_frame = 0
        self.descriptor_pools = []
        self.descriptor_writers = []

        self.device_num = None
        self.shader_stage_infos = []
        self.compute_shader_stage_info = None

        self.alive = True

        self.__command_buffers = []
        self.__compute_command_buffers = []

        self.__texture_images = []
        self.__texture_image_memories = []
        self.__texture_image_views = []
        self.__texture_samplers = []

        self.__descriptor_pool = None
        self.__uniform_buffers = []
        self._uniform_buffer_memories = []

        self._start_time = time.time()

        self.init_vulkan()

    def __del__(self):
        vkDeviceWaitIdle(self.vulkan_device.logical_device)

        for ts in self.__texture_samplers:
            vkDestroySampler(self.vulkan_device.logical_device, ts, None)

        for tiv in self.__texture_image_views:
            vkDestroyImageView(self.vulkan_device.logical_device, tiv, None)

        for ti in self.__texture_images:
            vkDestroyImage(self.vulkan_device.logical_device, ti, None)

        for tim in self.__texture_image_memories:
            vkFreeMemory(self.vulkan_device.logical_device, tim, None)

        if self.__descriptor_pool:
            vkDestroyDescriptorPool(self.vulkan_device.logical_device, self.__descriptor_pool, None)

        for ub in self.__uniform_buffers:
            vkDestroyBuffer(self.vulkan_device.logical_device, ub, None)

        for ubm in self._uniform_buffer_memories:
            vkFreeMemory(self.vulkan_device.logical_device, ubm, None)

        for shader in self.shader_stage_infos:
            vkDestroyShaderModule(self.vulkan_device.logical_device, shader.module, None)

        if self.compute_shader_stage_info is not None:
            vkDestroyShaderModule(self.vulkan_device.logical_device, self.compute_shader_stage_info.module, None)

        if self.vulkan_semaphores.image_available_semaphore:
            vkDestroySemaphore(
                self.vulkan_device.logical_device, self.vulkan_semaphores.image_available_semaphore, None
            )
        if self.vulkan_semaphores.render_finished_semaphore:
            vkDestroySemaphore(
                self.vulkan_device.logical_device, self.vulkan_semaphores.render_finished_semaphore, None
            )

        if self.vulkan_pipelines.descriptor_set_layout:
            vkDestroyDescriptorSetLayout(
                self.vulkan_device.logical_device, self.vulkan_pipelines.descriptor_set_layout, None
            )

        self.__cleanup_swap_chain()

        if self.vulkan_command_pool.pool:
            vkDestroyCommandPool(self.vulkan_device.logical_device, self.vulkan_command_pool.pool, None)

        if self.vulkan_device.logical_device:
            vkDestroyDevice(self.vulkan_device.logical_device, None)

        del self.vulkan_debugger

        del self.vulkan_surface

        if self.vulkan_instance.instance:
            vkDestroyInstance(self.vulkan_instance.instance, None)
            print("instance destroyed")

    def __cleanup_swap_chain(self):
        self.vulkan_resources.cleanup()
        [
            vkDestroyFramebuffer(self.vulkan_device.logical_device, i, None)
            for i in self.vulkan_resources.swap_chain_framebuffers
        ]
        self.vulkan_resources.swap_chain_framebuffers = []

        vkFreeCommandBuffers(
            self.vulkan_device.logical_device,
            self.vulkan_command_pool.pool,
            len(self.__command_buffers),
            self.__command_buffers,
        )
        self.vulkan_resources.swap_chain_framebuffers = []

        vkDestroyPipeline(self.vulkan_device.logical_device, self.vulkan_pipelines.pipeline, None)
        vkDestroyPipelineLayout(self.vulkan_device.logical_device, self.vulkan_pipelines.pipeline_layout, None)
        vkDestroyRenderPass(self.vulkan_device.logical_device, self.vulkan_renderer.render_pass, None)

        [
            vkDestroyImageView(self.vulkan_device.logical_device, i, None)
            for i in self.vulkan_renderer.swap_chain_image_views
        ]
        self.vulkan_renderer.swap_chain_image_views = []
        vkDestroySwapchainKHR(self.vulkan_device.logical_device, self.vulkan_swap.swap_chain, None)

    def recreate_swap_chain(self):
        vkDeviceWaitIdle(self.vulkan_device.logical_device)

        self.__cleanup_swap_chain()
        self.vulkan_swap.setup()
        self.vulkan_renderer.setup()

        self.vulkan_pipelines.reset_graphics()
        self.vulkan_resources.setup()
        self.vulkan_command_buffers.reset_graphics()

    def init_vulkan(self):
        self.vulkan_instance.setup()
        self.vulkan_debugger.setup()
        self.vulkan_surface.setup()
        self.vulkan_device.setup()
        self.vulkan_swap.setup()
        self.vulkan_renderer.setup()

        self.create_descriptor_set_layout()
        self.shader_stage_infos = self.create_shaders()
        self.compute_shader_stage_info = self.create_compute_shader()

        self.vulkan_pipelines.setup(self.shader_stage_infos, self.compute_shader_stage_info)

        self.vulkan_command_pool.setup()
        self.create_uniforms()

        self.vulkan_resources.setup()

        self.vulkan_descriptors.setup(self.descriptor_pools, self.descriptor_writers)

        self.vulkan_command_buffers.setup()

        self.vulkan_semaphores.setup()

    @abc.abstractmethod
    def create_descriptor_set_layout(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_shaders(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_compute_shader(self):
        raise NotImplementedError()

    def create_vertex_shader(self, shader_file="shader/vert.spv"):
        vertex_shader_mode = self._create_shader_module(shader_file)

        vertex_shader_stage_info = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_VERTEX_BIT, module=vertex_shader_mode, pName="main"
        )

        self.shader_stage_infos.append(vertex_shader_stage_info)

    def create_fragment_shader(self, shader_file="shader/frag.spv"):
        fragment_shader_mode = self._create_shader_module(shader_file)

        fragment_shader_stage_info = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_FRAGMENT_BIT, module=fragment_shader_mode, pName="main"
        )

        self.shader_stage_infos.append(fragment_shader_stage_info)

    def create_compute_shader_stage_info(self, shader_file="shader/comp.spv"):
        comp_shader_mode = self._create_shader_module(shader_file)

        compute_shader_stage_info = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT, module=comp_shader_mode, pName="main"
        )

        self.compute_shader_stage_info = compute_shader_stage_info

    def has_stencil_component(self, fm):
        return fm == VK_FORMAT_D32_SFLOAT_S8_UINT or fm == VK_FORMAT_D24_UNORM_S8_UINT

    def create_image_3d(self, widht, height, depth, im_format, tiling, usage, properties):
        image_info = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_3D,
            extent=[widht, height, depth],
            mipLevels=1,
            arrayLayers=1,
            format=im_format,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=tiling,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )

        image = vkCreateImage(self.vulkan_device.logical_device, image_info, None)

        mem_reuirements = vkGetImageMemoryRequirements(self.vulkan_device.logical_device, image)
        alloc_info = VkMemoryAllocateInfo(
            allocationSize=mem_reuirements.size,
            memoryTypeIndex=self.__find_memory_type(
                mem_reuirements.memoryTypeBits, properties
            ),
        )

        image_memory = vkAllocateMemory(self.vulkan_device.logical_device, alloc_info, None)

        vkBindImageMemory(self.vulkan_device.logical_device, image, image_memory, 0)

        return (image, image_memory)

    def create_image(self, widht, height, im_format, tiling, usage, properties):
        image_info = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D,
            extent=[widht, height, 1],
            mipLevels=1,
            arrayLayers=1,
            format=im_format,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=tiling,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )

        image = vkCreateImage(self.vulkan_device.logical_device, image_info, None)

        mem_reuirements = vkGetImageMemoryRequirements(self.vulkan_device.logical_device, image)
        alloc_info = VkMemoryAllocateInfo(
            allocationSize=mem_reuirements.size,
            memoryTypeIndex=self.__find_memory_type(
                mem_reuirements.memoryTypeBits, properties
            ),
        )

        image_memory = vkAllocateMemory(self.vulkan_device.logical_device, alloc_info, None)

        vkBindImageMemory(self.vulkan_device.logical_device, image, image_memory, 0)

        return (image, image_memory)

    def transition_image_layout(self, image, im_format, old_layout, new_layout, queue):
        cmd_buffer = self.__begin_single_time_commands()

        subresource_range = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )
        if new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            subresource_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT
            if self.has_stencil_component(im_format):
                subresource_range.aspectMask = (
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                )

        barrier = VkImageMemoryBarrier(
            oldLayout=old_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=subresource_range,
        )

        if (
                old_layout == VK_IMAGE_LAYOUT_UNDEFINED
                and new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        ):
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT
        elif (
                old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                and new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        ):
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT

            source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT
            destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        elif (
                old_layout == VK_IMAGE_LAYOUT_UNDEFINED
                and new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        ):
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = (
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            )

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
        else:
            raise SystemError("unsupported layout transition!")

        vkCmdPipelineBarrier(
            cmd_buffer, source_stage, destination_stage, 0, 0, None, 0, None, 1, barrier
        )

        self.__end_single_time_commands(cmd_buffer, queue)

    def __copy_buffer_to_image(self, buffer, image, width, height, depth=1, queue=None):
        cmdbuffer = self.__begin_single_time_commands()

        subresource = VkImageSubresourceLayers(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            mipLevel=0,
            baseArrayLayer=0,
            layerCount=1,
        )
        region = VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=subresource,
            imageOffset=None,
            imageExtent=[width, height, depth],
        )

        vkCmdCopyBufferToImage(
            cmdbuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, region
        )

        if queue is None:
            queue = self.vulkan_device.graphic_queue
        self.__end_single_time_commands(cmdbuffer, queue)

    def __create_buffer(self, size, usage, properties, require_heap=False):
        buffer_info = VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        vk_buffer = vkCreateBuffer(self.vulkan_device.logical_device, buffer_info, None)

        mem_requirements = vkGetBufferMemoryRequirements(self.vulkan_device.logical_device, vk_buffer)
        alloc_info = VkMemoryAllocateInfo(
            allocationSize=mem_requirements.size,
            memoryTypeIndex=self.__find_memory_type(
                mem_requirements.memoryTypeBits, properties, require_heap
            ),
        )
        buffer_memory = vkAllocateMemory(self.vulkan_device.logical_device, alloc_info, None)

        vkBindBufferMemory(self.vulkan_device.logical_device, vk_buffer, buffer_memory, 0)

        return (vk_buffer, buffer_memory)

    def __begin_single_time_commands(self):
        alloc_info = VkCommandBufferAllocateInfo(
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.vulkan_command_pool.pool,
            commandBufferCount=1,
        )

        command_buffer = vkAllocateCommandBuffers(self.vulkan_device.logical_device, alloc_info)[0]

        begin_info = VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vkBeginCommandBuffer(command_buffer, begin_info)

        return command_buffer

    def __end_single_time_commands(self, command_buffer, queue):
        vkEndCommandBuffer(command_buffer)

        submit_info = VkSubmitInfo(pCommandBuffers=[command_buffer])

        vkQueueSubmit(queue, 1, [submit_info], VK_NULL_HANDLE)
        vkQueueWaitIdle(queue)

        vkFreeCommandBuffers(
            self.vulkan_device.logical_device, self.vulkan_command_pool.pool, 1, [command_buffer]
        )

    def __copy_buffer(self, src, dst, buffer_size, queue):
        command_buffer = self.__begin_single_time_commands()

        copy_region = VkBufferCopy(0, 0, buffer_size)
        vkCmdCopyBuffer(command_buffer, src, dst, 1, [copy_region])

        self.__end_single_time_commands(command_buffer, queue)

    def __find_memory_type(self, type_filter, properties, require_heap=False):
        mem_properties = vkGetPhysicalDeviceMemoryProperties(self.vulkan_device.physical_device)

        for i, prop in enumerate(mem_properties.memoryTypes):
            if (type_filter & (1 << i)) and (
                    (prop.propertyFlags & properties) == properties
            ) and ((not require_heap) or (
                    mem_properties.memoryHeaps[prop.heapIndex].flags
                    & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
            )):
                return i

        return -1

    @abc.abstractmethod
    def update_uniform_buffers(self):
        raise NotImplementedError()

    def dispatch_compute(self):
        # Bind the compute pipeline, descriptor sets, and dispatch parameters
        compute_submit_info = VkSubmitInfo(
            waitSemaphoreCount=1,  # Wait for the write semaphore
            pWaitSemaphores=[self.vulkan_semaphores.write_semaphore],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT],
            commandBufferCount=1,
            pCommandBuffers=self.__compute_command_buffers,
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.vulkan_semaphores.read_semaphore]
        )
        vkQueueSubmit(self.vulkan_device.compute_queue, 1, compute_submit_info, VK_NULL_HANDLE)

    def draw_frame(self):
        try:
            image_index = vkAcquireNextImageKHR(
                self.vulkan_device.logical_device,
                self.vulkan_swap.swap_chain,
                18446744073709551615,
                self.vulkan_semaphores.image_available_semaphore,
                VK_NULL_HANDLE,
            )
        except VkErrorSurfaceLostKhr:
            self.recreate_swap_chain()
            return

        wait_semaphores = [self.vulkan_semaphores.image_available_semaphore]
        signal_semaphores = [self.vulkan_semaphores.render_finished_semaphore]
        wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        submit = VkSubmitInfo(
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=wait_stages,
            pCommandBuffers=[self.__command_buffers[image_index]],
            pSignalSemaphores=signal_semaphores,
        )

        vkQueueSubmit(self.vulkan_device.graphic_queue, 1, submit, VK_NULL_HANDLE)

        present_info = VkPresentInfoKHR(
            pWaitSemaphores=signal_semaphores,
            pSwapchains=[self.vulkan_swap.swap_chain],
            pImageIndices=[image_index],
        )

        try:
            vkQueuePresentKHR(self.vulkan_device.present_queue, present_info)
        except VkErrorOutOfDateKhr:
            self.recreate_swap_chain()

        if self.enable_validation_layers:
            vkQueueWaitIdle(self.vulkan_device.present_queue)

    def _create_shader_module(self, shader_file):
        with open(shader_file, "rb") as sf:
            code = sf.read()

            create_info = VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)

            return vkCreateShaderModule(self.vulkan_device.logical_device, create_info, None)

    @abc.abstractmethod
    def create_uniforms(self):
        raise NotImplementedError()

    def create_3d_texture(self, npz_file, binding, vkformat=VK_FORMAT_R8_SINT, is_compute=False, is_compute_view=False):
        _image = np.load(npz_file)
        _image = _image[_image.files[0]]
        shape = _image.shape
        width = _image.shape[0]
        height = _image.shape[1]
        depth = _image.shape[2]
        if vkformat == VK_FORMAT_R8G8B8A8_UNORM:
            _image = _image.astype(np.uint8)
            image_size = width * height * depth * 4
        elif vkformat in [VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT, VK_FORMAT_R8_UNORM]:
            _image = _image.astype(np.uint8)
            image_size = width * height * depth
        else:
            raise NotImplementedError(f"pls fix code for vkformat: {(vkformat)} (and search for the right name)")

        if is_compute_view:
            staging_buffer, staging_mem = self.__create_buffer(
                image_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                require_heap=True
            )
        else:
            staging_buffer, staging_mem = self.__create_buffer(
                image_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )

        data = vkMapMemory(self.vulkan_device.logical_device, staging_mem, 0, image_size, 0)
        ffi.memmove(data, _image.tobytes(), image_size)
        vkUnmapMemory(self.vulkan_device.logical_device, staging_mem)

        del _image

        texture_image, texture_image_memory = self.create_image_3d(
            width,
            height,
            depth,
            vkformat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        self.__texture_images.append(texture_image)
        self.__texture_image_memories.append(texture_image_memory)

        if is_compute or is_compute_view:
            queue = self.vulkan_device.compute_queue
        else:
            queue = self.vulkan_device.graphic_queue

        self.transition_image_layout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            queue
        )
        self.__copy_buffer_to_image(staging_buffer, texture_image, width, height, depth, queue)
        self.transition_image_layout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            queue
        )

        vkDestroyBuffer(self.vulkan_device.logical_device, staging_buffer, None)
        vkFreeMemory(self.vulkan_device.logical_device, staging_mem, None)

        self.__texture_image_views.append(self.vulkan_renderer.create_image_view(
            self.__texture_images[-1], vkformat, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_3D
        ))

        sampler_info = VkSamplerCreateInfo(
            magFilter=VK_FILTER_LINEAR,
            minFilter=VK_FILTER_LINEAR,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            anisotropyEnable=True,
            maxAnisotropy=16,
            compareEnable=False,
            compareOp=VK_COMPARE_OP_ALWAYS,
            borderColor=VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalizedCoordinates=False,
        )

        self.__texture_samplers.append(vkCreateSampler(
            self.vulkan_device.logical_device, sampler_info, None
        ))

        image_info = VkDescriptorImageInfo(
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            imageView=self.__texture_image_views[-1],
            sampler=self.__texture_samplers[-1],
        )

        pool_size2 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1
        )

        self.descriptor_pools.append(pool_size2)

        self.descriptor_writers.append(lambda x: VkWriteDescriptorSet(
            dstSet=x,
            dstBinding=binding,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo=[image_info],
        ))

        return shape

    def create_2d_texture(self, file, binding, vkformat=VK_FORMAT_R8G8B8A8_UNORM, is_compute=False,
                          is_compute_view=False):
        _image = Image.open(file)
        _image.putalpha(1)
        width = _image.width
        height = _image.height
        if vkformat == VK_FORMAT_R8G8B8A8_UNORM:
            image_size = width * height * 4
            shape = [width, height, 4]
        else:
            raise NotImplementedError(f"pls fix code for vkformat: {(vkformat)} (and search for the right name)")

        if is_compute_view:
            staging_buffer, staging_mem = self.__create_buffer(
                image_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                require_heap=True
            )
        else:
            staging_buffer, staging_mem = self.__create_buffer(
                image_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )

        data = vkMapMemory(self.vulkan_device.logical_device, staging_mem, 0, image_size, 0)
        ffi.memmove(data, _image.tobytes(), image_size)
        vkUnmapMemory(self.vulkan_device.logical_device, staging_mem)

        del _image

        texture_image, texture_image_memory = self.create_image(
            width,
            height,
            vkformat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        self.__texture_images.append(texture_image)
        self.__texture_image_memories.append(texture_image_memory)

        if is_compute or is_compute_view:
            queue = self.vulkan_device.compute_queue
        else:
            queue = self.vulkan_device.graphic_queue
        self.transition_image_layout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            queue
        )
        self.__copy_buffer_to_image(staging_buffer, texture_image, width, height, queue=queue)
        self.transition_image_layout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            queue
        )

        vkDestroyBuffer(self.vulkan_device.logical_device, staging_buffer, None)
        vkFreeMemory(self.vulkan_device.logical_device, staging_mem, None)

        self.__texture_image_views.append(self.vulkan_renderer.create_image_view(
            self.__texture_images[-1], vkformat, VK_IMAGE_ASPECT_COLOR_BIT
        ))

        sampler_info = VkSamplerCreateInfo(
            magFilter=VK_FILTER_LINEAR,
            minFilter=VK_FILTER_LINEAR,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT,
            anisotropyEnable=True,
            maxAnisotropy=16,
            compareEnable=False,
            compareOp=VK_COMPARE_OP_ALWAYS,
            borderColor=VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalizedCoordinates=False,
        )

        self.__texture_samplers.append(vkCreateSampler(
            self.vulkan_device.logical_device, sampler_info, None
        ))

        image_info = VkDescriptorImageInfo(
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            imageView=self.__texture_image_views[-1],
            sampler=self.__texture_samplers[-1],
        )

        pool_size2 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1
        )

        self.descriptor_pools.append(pool_size2)

        self.descriptor_writers.append(lambda x: VkWriteDescriptorSet(
            dstSet=x,
            dstBinding=binding,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo=[image_info],
        ))

        return shape

    def create_ubo(self, ubo, binding, is_compute_view=False):
        if is_compute_view:
            __uniform_buffer, __uniform_buffer_memory = self.__create_buffer(
                ubo.nbytes,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                require_heap=True
            )
        else:
            __uniform_buffer, __uniform_buffer_memory = self.__create_buffer(
                ubo.nbytes,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )

        self.__uniform_buffers.append(__uniform_buffer)
        self._uniform_buffer_memories.append(__uniform_buffer_memory)

        buffer_info = VkDescriptorBufferInfo(
            buffer=__uniform_buffer, offset=0, range=ubo.nbytes
        )

        pool_size1 = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1
        )

        self.descriptor_pools.append(pool_size1)

        self.descriptor_writers.append(lambda x: VkWriteDescriptorSet(
            dstSet=x,
            dstBinding=binding,
            dstArrayElement=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=[buffer_info],
        ))

    def render(self):
        self.update_uniform_buffers()
        self.draw_frame()

    def render_once(self):
        self.sdl2_event_check()
        self.render()

    @abc.abstractmethod
    def sdl2_event_check(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_compute_ubos(self, data):
        raise NotImplementedError()

    @abc.abstractmethod
    def read_compute_ubos(self):
        raise NotImplementedError()

    def compute_once(self, data):
        self.write_compute_ubos(data)
        self.dispatch_compute()
        data = self.read_compute_ubos()
        return data


class AbstractUBO(object):
    @abc.abstractmethod
    def to_bytes(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def nbytes(self):
        raise NotImplementedError()
