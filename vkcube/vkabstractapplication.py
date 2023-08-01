import abc
import ctypes
import sys
import time
from collections import defaultdict

import numpy as np
from . import sdl_include
import sdl2
import sdl2.ext
from PIL import Image
from vulkan import (
    VK_ERROR_EXTENSION_NOT_PRESENT,
    vkGetInstanceProcAddr,
    vkGetDeviceProcAddr,
    VkVertexInputBindingDescription,
    VK_VERTEX_INPUT_RATE_VERTEX,
    VkVertexInputAttributeDescription,
    VK_FORMAT_R32G32B32_SFLOAT,
    VK_FORMAT_R32G32_SFLOAT,
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
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
    VkApplicationInfo,
    VK_STRUCTURE_TYPE_APPLICATION_INFO,
    VK_MAKE_VERSION,
    VK_API_VERSION,
    vkEnumerateInstanceLayerProperties,
    VkInstanceCreateInfo,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    vkCreateInstance,
    VkDebugReportCallbackCreateInfoEXT,
    VK_DEBUG_REPORT_WARNING_BIT_EXT,
    VK_DEBUG_REPORT_ERROR_BIT_EXT,
    VkXlibSurfaceCreateInfoKHR,
    VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
    VkWaylandSurfaceCreateInfoKHR,
    VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
    VkWin32SurfaceCreateInfoKHR,
    VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
    vkEnumeratePhysicalDevices,
    VkDeviceQueueCreateInfo,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    VkPhysicalDeviceFeatures,
    VkDeviceCreateInfo,
    vkCreateDevice,
    vkGetDeviceQueue,
    VkSwapchainCreateInfoKHR,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VK_SHARING_MODE_CONCURRENT,
    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_IMAGE_ASPECT_COLOR_BIT,
    VkAttachmentDescription,
    VK_SAMPLE_COUNT_1_BIT,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_DONT_CARE,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VkAttachmentReference,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VkSubpassDescription,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    VkSubpassDependency,
    VK_SUBPASS_EXTERNAL,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    VkRenderPassCreateInfo,
    vkCreateRenderPass,
    VkPipelineShaderStageCreateInfo,
    VK_SHADER_STAGE_VERTEX_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT,
    VkPipelineVertexInputStateCreateInfo,
    VkPipelineInputAssemblyStateCreateInfo,
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    VkViewport,
    VkRect2D,
    VkPipelineViewportStateCreateInfo,
    VkPipelineRasterizationStateCreateInfo,
    VK_POLYGON_MODE_FILL,
    VK_CULL_MODE_BACK_BIT,
    VK_FRONT_FACE_CLOCKWISE,
    VkPipelineMultisampleStateCreateInfo,
    VkPipelineDepthStencilStateCreateInfo,
    VK_COMPARE_OP_LESS,
    VkPipelineColorBlendAttachmentState,
    VK_COLOR_COMPONENT_R_BIT,
    VK_COLOR_COMPONENT_G_BIT,
    VK_COLOR_COMPONENT_B_BIT,
    VK_COLOR_COMPONENT_A_BIT,
    VkPipelineColorBlendStateCreateInfo,
    VK_LOGIC_OP_COPY,
    VkPipelineLayoutCreateInfo,
    vkCreatePipelineLayout,
    VkGraphicsPipelineCreateInfo,
    VK_NULL_HANDLE,
    vkCreateGraphicsPipelines,
    vkDestroyShaderModule,
    VkFramebufferCreateInfo,
    vkCreateFramebuffer,
    VkCommandPoolCreateInfo,
    vkCreateCommandPool,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    vkGetPhysicalDeviceFormatProperties,
    VK_IMAGE_TILING_LINEAR,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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
    VkImageViewCreateInfo,
    VK_IMAGE_VIEW_TYPE_2D,
    vkCreateImageView,
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
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VkDescriptorPoolSize,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VkDescriptorPoolCreateInfo,
    vkCreateDescriptorPool,
    VkDescriptorSetAllocateInfo,
    vkAllocateDescriptorSets,
    VkDescriptorBufferInfo,
    VkDescriptorImageInfo,
    VkWriteDescriptorSet,
    vkUpdateDescriptorSets,
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
    VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    VkClearValue,
    VkRenderPassBeginInfo,
    vkCmdBeginRenderPass,
    VK_SUBPASS_CONTENTS_INLINE,
    vkCmdBindPipeline,
    vkCmdBindVertexBuffers,
    vkCmdBindIndexBuffer,
    VK_INDEX_TYPE_UINT32,
    vkCmdBindDescriptorSets,
    vkCmdDrawIndexed,
    vkCmdEndRenderPass,
    VkSemaphoreCreateInfo,
    vkCreateSemaphore,
    VkErrorSurfaceLostKhr,
    VkPresentInfoKHR,
    VkErrorOutOfDateKhr,
    VkShaderModuleCreateInfo,
    vkCreateShaderModule,
    VK_FORMAT_UNDEFINED,
    VK_FORMAT_B8G8R8_UNORM,
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    VK_PRESENT_MODE_FIFO_KHR,
    VK_PRESENT_MODE_MAILBOX_KHR,
    VK_PRESENT_MODE_IMMEDIATE_KHR,
    VkExtent2D,
    vkGetPhysicalDeviceFeatures,
    vkEnumerateDeviceExtensionProperties,
    vkGetPhysicalDeviceQueueFamilyProperties,
    VK_QUEUE_GRAPHICS_BIT,
    vkEnumerateInstanceExtensionProperties,
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_IMAGE_TYPE_3D, VK_FORMAT_R8_SINT, VK_IMAGE_VIEW_TYPE_3D, vkCmdDraw,
)

class InstanceProcAddr(object):
    """Dynamically check for and load function from Vulkan"""

    T = None

    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        funcName = self.__func.__name__
        func = InstanceProcAddr.procfunc(funcName)
        if func:
            return func(*args, **kwargs)
        else:
            return VK_ERROR_EXTENSION_NOT_PRESENT

    @staticmethod
    def procfunc(funcName):
        return vkGetInstanceProcAddr(InstanceProcAddr.T, funcName)


class DeviceProcAddr(InstanceProcAddr):
    """Gets function addresses specific to devices rather than instances."""

    @staticmethod
    def procfunc(funcName):
        return vkGetDeviceProcAddr(InstanceProcAddr.T, funcName)


@InstanceProcAddr
def vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroyDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkCreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroySurfaceKHR(instance, surface, pAllocator):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface):
    pass


@DeviceProcAddr
def vkCreateSwapchainKHR(device, pCreateInfo, pAllocator):
    pass


@DeviceProcAddr
def vkDestroySwapchainKHR(device, swapchain, pAllocator):
    pass


@DeviceProcAddr
def vkGetSwapchainImagesKHR(device, swapchain):
    pass


@DeviceProcAddr
def vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence):
    pass


@DeviceProcAddr
def vkQueuePresentKHR(queue, pPresentInfo):
    pass


def debugCallback(*args):
    print("DEBUG: {} {}".format(args[5], args[6]))
    return 0


class Win32misc(object):
    @staticmethod
    def getInstance(hWnd):
        from cffi import FFI as _FFI

        _ffi = _FFI()
        _ffi.cdef("long __stdcall GetWindowLongA(void* hWnd, int nIndex);")
        _lib = _ffi.dlopen("User32.dll")
        return _lib.GetWindowLongA(_ffi.cast("void*", hWnd), -6)  # GWL_HINSTANCE


class QueueFamilyIndices(object):
    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    @property
    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


class SwapChainSupportDetails(object):
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None


class Vertex(object):
    POS = np.array([0, 0, 0], np.float32)
    COLOR = np.array([0, 0, 0], np.float32)
    TEXCOORD = np.array([0, 0], np.float32)

    # def __init__(self):
    #     self.pos = []
    #     self.color = []

    @staticmethod
    def getBindingDescription():
        bindingDescription = VkVertexInputBindingDescription(
            binding=0,
            stride=Vertex.POS.nbytes + Vertex.COLOR.nbytes + Vertex.TEXCOORD.nbytes,
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX,
        )

        return bindingDescription

    @staticmethod
    def getAttributeDescriptions():
        pos = VkVertexInputAttributeDescription(
            location=0, binding=0, format=VK_FORMAT_R32G32B32_SFLOAT, offset=0
        )

        color = VkVertexInputAttributeDescription(
            location=1,
            binding=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=Vertex.POS.nbytes,
        )

        texcoord = VkVertexInputAttributeDescription(
            location=2,
            binding=0,
            format=VK_FORMAT_R32G32_SFLOAT,
            offset=Vertex.POS.nbytes + Vertex.COLOR.nbytes,
        )
        return [pos, color, texcoord]


class UniformBufferObject(object):
    def __init__(self):
        self.model = np.identity(4, np.float32)
        self.view = np.identity(4, np.float32)
        self.proj = np.identity(4, np.float32)

    def to_array(self):
        return np.concatenate((self.model, self.view, self.proj))

    @property
    def nbytes(self):
        return self.proj.nbytes + self.view.nbytes + self.model.nbytes


class VKAbstractApplication:
    def __init__(self, width=1280, height=720, enable_validation_layers=True):
        super(VKAbstractApplication, self).__init__()

        self.descriptor_pools = []
        self.descriptor_writers = []

        self.enable_validation_layers = enable_validation_layers
        self.width = width
        self.height = height
        self.window, self.wm_info = self.init_sdl_window()
        sdl2.SDL_SetWindowTitle(self.window, ctypes.c_char_p(b"VKCube"))
        self.device_num = None
        self.device_extensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        self.shader_stage_infos = []
        self.validation_layers = []

        self.event_dict = dict()
        self.event_dict[sdl2.SDL_WINDOWEVENT_RESIZED] = self.resize_event

        self.__instance = None
        self.__callbcak = None
        self.__surface = None

        self.__physical_device = None
        self._logical_device = None
        self.__graphic_queue = None
        self.__present_queue = None

        self.__swap_chain = None
        self.__swap_chain_images = []
        self.__swap_chain_image_format = None
        self.__swap_chain_extent = None
        self.__swap_chain_image_views = []
        self.__swap_chain_framebuffers = []

        self.__render_pass = None
        self.__pipeline = None
        self.__pipeline_layout = None

        self.__command_pool = None
        self.__command_buffers = []

        self.__image_available_semaphore = None
        self.__render_finished_semaphore = None

        self.__texture_images = []
        self.__texture_image_memories = []
        self.__texture_image_views = []
        self.__texture_samplers = []

        self.__depth_image = None
        self.__depth_image_memory = None
        self.__depth_image_view = None

        self.__descriptor_pool = None
        self.__descriptor_set = None
        self._descriptor_set_layout = None
        self.__uniform_buffers = []
        self._uniform_buffer_memories = []

        self.__ubo = UniformBufferObject()

        self._start_time = time.time()

        self.init_vulkan()

    def init_sdl_window(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError(sdl2.SDL_GetError())

        window = sdl2.SDL_CreateWindow(
            "test".encode("ascii"),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_VULKAN | sdl2.SDL_WINDOW_RESIZABLE,
        )

        if not window:
            raise RuntimeError(sdl2.SDL_GetError())

        wm_info = sdl2.SDL_SysWMinfo()
        sdl2.SDL_VERSION(wm_info.version)
        sdl2.SDL_GetWindowWMInfo(window, ctypes.byref(wm_info))
        return window, wm_info

    def get_window_size(self):
        w = ctypes.c_int()
        h = ctypes.c_int()
        sdl2.SDL_GetWindowSize(self.window, w, h)
        return w.value, h.value

    def get_mouse(self):
        x, y, s = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_uint32(0)  # Create two ctypes values
        # Pass x and y as references (pointers) to SDL_GetMouseState()
        buttonstate = sdl2.mouse.SDL_GetMouseState(ctypes.byref(x), ctypes.byref(y))
        # Print x and y as Python values
        return x.value, y.value, buttonstate

    def __del__(self):
        vkDeviceWaitIdle(self._logical_device)

        for ts in self.__texture_samplers:
            vkDestroySampler(self._logical_device, ts, None)

        for tiv in self.__texture_image_views:
            vkDestroyImageView(self._logical_device, tiv, None)

        for ti in self.__texture_images:
            vkDestroyImage(self._logical_device, ti, None)

        for tim in self.__texture_image_memories:
            vkFreeMemory(self._logical_device, tim, None)

        if self.__descriptor_pool:
            vkDestroyDescriptorPool(self._logical_device, self.__descriptor_pool, None)

        for ub in self.__uniform_buffers:
            vkDestroyBuffer(self._logical_device, ub, None)

        for ubm in self._uniform_buffer_memories:
            vkFreeMemory(self._logical_device, ubm, None)

        for shader in self.shader_stage_infos:
            vkDestroyShaderModule(self._logical_device, shader.module, None)

        '''if self.__vertex_buffer:
            vkDestroyBuffer(self._logical_device, self.__vertex_buffer, None)

        if self.__vertex_buffer_memory:
            vkFreeMemory(self._logical_device, self.__vertex_buffer_memory, None)

        if self.__index_buffer:
            vkDestroyBuffer(self._logical_device, self.__index_buffer, None)

        if self.__index_buffer_memory:
            vkFreeMemory(self._logical_device, self.__index_buffer_memory, None)'''

        if self.__image_available_semaphore:
            vkDestroySemaphore(
                self._logical_device, self.__image_available_semaphore, None
            )
        if self.__render_finished_semaphore:
            vkDestroySemaphore(
                self._logical_device, self.__render_finished_semaphore, None
            )

        if self._descriptor_set_layout:
            vkDestroyDescriptorSetLayout(
                self._logical_device, self._descriptor_set_layout, None
            )

        self.__cleanup_swap_chain()

        if self.__command_pool:
            vkDestroyCommandPool(self._logical_device, self.__command_pool, None)

        if self._logical_device:
            vkDestroyDevice(self._logical_device, None)

        if self.__callbcak:
            vkDestroyDebugReportCallbackEXT(self.__instance, self.__callbcak, None)

        if self.__surface:
            vkDestroySurfaceKHR(self.__instance, self.__surface, None)

        if self.__instance:
            vkDestroyInstance(self.__instance, None)
            print("instance destroyed")

    def __cleanup_swap_chain(self):
        vkDestroyImageView(self._logical_device, self.__depth_image_view, None)
        vkDestroyImage(self._logical_device, self.__depth_image, None)
        vkFreeMemory(self._logical_device, self.__depth_image_memory, None)

        [
            vkDestroyFramebuffer(self._logical_device, i, None)
            for i in self.__swap_chain_framebuffers
        ]
        self.__swap_chain_framebuffers = []

        vkFreeCommandBuffers(
            self._logical_device,
            self.__command_pool,
            len(self.__command_buffers),
            self.__command_buffers,
        )
        self.__swap_chain_framebuffers = []

        vkDestroyPipeline(self._logical_device, self.__pipeline, None)
        vkDestroyPipelineLayout(self._logical_device, self.__pipeline_layout, None)
        vkDestroyRenderPass(self._logical_device, self.__render_pass, None)

        [
            vkDestroyImageView(self._logical_device, i, None)
            for i in self.__swap_chain_image_views
        ]
        self.__swap_chain_image_views = []
        vkDestroySwapchainKHR(self._logical_device, self.__swap_chain, None)

    def __recreate_swap_chain(self):
        vkDeviceWaitIdle(self._logical_device)

        self.__cleanup_swap_chain()
        self.__create_swap_chain()
        self.__create_image_views()
        self.__create_render_pass()

        self.__create_graphics_pipeline()
        self.__create_depth_resources()
        self.__create_frambuffers()
        self.__create_command_buffers()

    def init_vulkan(self):
        self._create_instance()
        self._setup_debug_callback()
        self._create_surface()
        self._pick_physical_device()
        self.__create_logical_device()
        self.__create_swap_chain()
        self.__create_image_views()
        self.__create_render_pass()

        self.create_descriptor_set_layout()
        self.shader_stage_infos = self.create_shaders()

        self.__create_command_pool()
        self.create_uniforms()

        self.__create_graphics_pipeline()
        self.__create_depth_resources()
        self.__create_frambuffers()
        self._create_descriptors()
        self.__create_command_buffers()
        self.__create_semaphores()

    def _create_instance(self):
        appInfo = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Python VKCube",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="pyvulkan",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION,
        )

        layers = vkEnumerateInstanceLayerProperties()
        layers = [l.layerName for l in layers]
        print("availables layers: %s\n" % layers)

        if self.enable_validation_layers:
            if "VK_LAYER_KHRONOS_validation" in layers:
                self.validation_layers = ["VK_LAYER_KHRONOS_validation"]
            elif "VK_LAYER_LUNARG_standard_validation" in layers:
                self.validation_layers = ["VK_LAYER_LUNARG_standard_validation"]
            else:
                raise SystemError("validation layers requested, but not available!")
        else:
            self.validation_layers = []

        extenstions = self.__getRequiredExtensions()
        if self.enable_validation_layers:
            instanceInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                flags=0,
                pApplicationInfo=appInfo,
                enabledLayerCount=len(self.validation_layers),
                ppEnabledLayerNames=self.validation_layers,
                enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions,
            )
        else:
            instanceInfo = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                flags=0,
                pApplicationInfo=appInfo,
                enabledLayerCount=0,
                enabledExtensionCount=len(extenstions),
                ppEnabledExtensionNames=extenstions,
            )

        self.__instance = vkCreateInstance(instanceInfo, None)

        InstanceProcAddr.T = self.__instance

    def _setup_debug_callback(self):
        if not self.enable_validation_layers:
            return

        createInfo = VkDebugReportCallbackCreateInfoEXT(
            flags=VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT,
            pfnCallback=debugCallback,
        )

        self.__callbcak = vkCreateDebugReportCallbackEXT(
            self.__instance, createInfo, None
        )

    def _create_surface(self):
        def surface_xlib():
            print("Create Xlib surface")
            vkCreateXlibSurfaceKHR = vkGetInstanceProcAddr(
                self.__instance, "vkCreateXlibSurfaceKHR"
            )
            surface_create = VkXlibSurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                dpy=self.wm_info.info.x11.display,
                window=self.wm_info.info.x11.window,
                flags=0,
            )
            return vkCreateXlibSurfaceKHR(self.__instance, surface_create, None)

        def surface_wayland():
            print("Create wayland surface")
            vkCreateWaylandSurfaceKHR = vkGetInstanceProcAddr(
                self.__instance, "vkCreateWaylandSurfaceKHR"
            )
            surface_create = VkWaylandSurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                display=self.wm_info.info.wl.display,
                surface=self.wm_info.info.wl.surface,
                flags=0,
            )
            return vkCreateWaylandSurfaceKHR(self.__instance, surface_create, None)

        def surface_win32():
            def get_instance(hWnd):
                """Hack needed before SDL 2.0.6"""
                from cffi import FFI

                _ffi = FFI()
                _ffi.cdef("long __stdcall GetWindowLongA(void* hWnd, int nIndex);")
                _lib = _ffi.dlopen("User32.dll")
                return _lib.GetWindowLongA(_ffi.cast("void*", hWnd), -6)

            print("Create windows surface")
            vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(
                self.__instance, "vkCreateWin32SurfaceKHR"
            )
            surface_create = VkWin32SurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                hinstance=get_instance(self.wm_info.info.win.window),
                hwnd=self.wm_info.info.win.window,
                flags=0,
            )
            return vkCreateWin32SurfaceKHR(self.__instance, surface_create, None)

        surface_mapping = {
            sdl2.SDL_SYSWM_X11: surface_xlib,
            sdl2.SDL_SYSWM_WAYLAND: surface_wayland,
            sdl2.SDL_SYSWM_WINDOWS: surface_win32,
        }

        self.__surface = surface_mapping[self.wm_info.subsystem]()

    def _pick_physical_device(self):
        physicalDevices = vkEnumeratePhysicalDevices(self.__instance)

        for e, device in enumerate(physicalDevices):
            if self.__isDeviceSuitable(device):
                self.__physical_device = device
                self.device_num = e  # use this to ensure other GPU, if any, is used for compute shaders
                break

        assert self.__physical_device != None

    def __create_logical_device(self):
        indices = self.__findQueueFamilies(self.__physical_device)

        uniqueQueueFamilies = {}.fromkeys(
            [indices.graphicsFamily, indices.presentFamily]
        )
        queueCreateInfos = []
        for i in uniqueQueueFamilies:
            queueCreateInfo = VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=i,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            queueCreateInfos.append(queueCreateInfo)

        deviceFeatures = VkPhysicalDeviceFeatures()
        deviceFeatures.samplerAnisotropy = True
        if self.enable_validation_layers:
            createInfo = VkDeviceCreateInfo(
                queueCreateInfoCount=len(queueCreateInfos),
                pQueueCreateInfos=queueCreateInfos,
                enabledExtensionCount=len(self.device_extensions),
                ppEnabledExtensionNames=self.device_extensions,
                enabledLayerCount=len(self.validation_layers),
                ppEnabledLayerNames=self.validation_layers,
                pEnabledFeatures=deviceFeatures,
            )
        else:
            createInfo = VkDeviceCreateInfo(
                queueCreateInfoCount=1,
                pQueueCreateInfos=queueCreateInfo,
                enabledExtensionCount=len(self.device_extensions),
                ppEnabledExtensionNames=self.device_extensions,
                enabledLayerCount=0,
                pEnabledFeatures=deviceFeatures,
            )

        self._logical_device = vkCreateDevice(self.__physical_device, createInfo, None)

        DeviceProcAddr.T = self._logical_device

        self.__graphic_queue = vkGetDeviceQueue(
            self._logical_device, indices.graphicsFamily, 0
        )
        self.__present_queue = vkGetDeviceQueue(
            self._logical_device, indices.presentFamily, 0
        )

    def __create_swap_chain(self):
        swapChainSupport = self.__querySwapChainSupport(self.__physical_device)

        surfaceFormat = self.__chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = self.__chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = self.__chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if (
                swapChainSupport.capabilities.maxImageCount > 0
                and imageCount > swapChainSupport.capabilities.maxImageCount
        ):
            imageCount = swapChainSupport.capabilities.maxImageCount

        indices = self.__findQueueFamilies(self.__physical_device)
        queueFamily = {}.fromkeys([indices.graphicsFamily, indices.presentFamily])
        queueFamilies = list(queueFamily.keys())
        if len(queueFamilies) > 1:
            createInfo = VkSwapchainCreateInfoKHR(
                surface=self.__surface,
                minImageCount=imageCount,
                imageFormat=surfaceFormat.format,
                imageColorSpace=surfaceFormat.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                # queueFamilyIndexCount=len(queueFamilies),
                pQueueFamilyIndices=queueFamilies,
                imageSharingMode=VK_SHARING_MODE_CONCURRENT,
                preTransform=swapChainSupport.capabilities.currentTransform,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=presentMode,
                clipped=True,
            )
        else:
            createInfo = VkSwapchainCreateInfoKHR(
                surface=self.__surface,
                minImageCount=imageCount,
                imageFormat=surfaceFormat.format,
                imageColorSpace=surfaceFormat.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                # queueFamilyIndexCount=len(queueFamilies),
                pQueueFamilyIndices=queueFamilies,
                imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
                preTransform=swapChainSupport.capabilities.currentTransform,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=presentMode,
                clipped=True,
            )

        self.__swap_chain = vkCreateSwapchainKHR(self._logical_device, createInfo, None)
        assert self.__swap_chain != None

        self.__swap_chain_images = vkGetSwapchainImagesKHR(
            self._logical_device, self.__swap_chain
        )

        self.__swap_chain_image_format = surfaceFormat.format
        self.__swap_chain_extent = extent

    def __create_image_views(self):
        self.__swap_chain_image_views = []

        for i, image in enumerate(self.__swap_chain_images):
            self.__swap_chain_image_views.append(
                self.__createImageView(
                    image, self.__swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT
                )
            )

    def __create_render_pass(self):
        colorAttachment = VkAttachmentDescription(
            format=self.__swap_chain_image_format,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )

        depthAttachment = VkAttachmentDescription(
            format=self.depthFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

        colorAttachmentRef = VkAttachmentReference(
            0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        depthAttachmentRef = VkAttachmentReference(
            1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        subpass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            pColorAttachments=[colorAttachmentRef],
            pDepthStencilAttachment=[depthAttachmentRef],
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

        renderPassInfo = VkRenderPassCreateInfo(
            pAttachments=[colorAttachment, depthAttachment],
            pSubpasses=[subpass],
            pDependencies=[dependency],
        )

        self.__render_pass = vkCreateRenderPass(
            self._logical_device, renderPassInfo, None
        )

    @abc.abstractmethod
    def create_descriptor_set_layout(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_shaders(self):
        raise NotImplementedError()

    def create_vertex_shader(self, shader_file="shader/vert.spv"):
        vertexShaderMode = self._create_shader_module(shader_file)

        vertexShaderStageInfo = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_VERTEX_BIT, module=vertexShaderMode, pName="main"
        )

        self.shader_stage_infos.append(vertexShaderStageInfo)

    def create_fragment_shader(self, shader_file="shader/frag.spv"):
        fragmentShaderMode = self._create_shader_module(shader_file)

        fragmentShaderStageInfo = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_FRAGMENT_BIT, module=fragmentShaderMode, pName="main"
        )

        self.shader_stage_infos.append(fragmentShaderStageInfo)

    def __create_graphics_pipeline(self):
        bindingDescription = Vertex.getBindingDescription()
        attributeDescription = Vertex.getAttributeDescriptions()

        vertexInputInfo = VkPipelineVertexInputStateCreateInfo(
            # vertexBindingDescriptionCount=0,
            pVertexBindingDescriptions=None,
            # vertexAttributeDescriptionCount=0,
            pVertexAttributeDescriptions=None,
        )

        input_assembly = VkPipelineInputAssemblyStateCreateInfo(
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, primitiveRestartEnable=False
        )

        viewport = VkViewport(
            0.0,
            0.0,
            float(self.__swap_chain_extent.width),
            float(self.__swap_chain_extent.height),
            0.0,
            1.0,
        )

        scissor = VkRect2D([0, 0], self.__swap_chain_extent)
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
            # setLayoutCount=0,
            pushConstantRangeCount=0,
            pSetLayouts=[self._descriptor_set_layout],
        )

        self.__pipeline_layout = vkCreatePipelineLayout(
            self._logical_device, pipeline_layout_info, None
        )

        if len(self.shader_stage_infos) == 0:
            raise RuntimeError("Please create shaders before running graphics")

        pipeline_info = VkGraphicsPipelineCreateInfo(
            # stageCount=len(shaderStageInfos),
            pStages=self.shader_stage_infos,
            pVertexInputState= vertexInputInfo,
            pInputAssemblyState= input_assembly,
            pViewportState=viewport_stage,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=color_bending,
            pDepthStencilState=depth_stencil,
            layout=self.__pipeline_layout,
            renderPass=self.__render_pass,
            subpass=0,
            basePipelineHandle=VK_NULL_HANDLE,
        )

        self.__pipeline = vkCreateGraphicsPipelines(
            self._logical_device, VK_NULL_HANDLE, 1, pipeline_info, None
        )[0]

        #for shader in self.shader_stage_infos:
        #    vkDestroyShaderModule(self._logical_device, shader.module, None)

    def __create_frambuffers(self):
        self.__swap_chain_framebuffers = []
        for i, iv in enumerate(self.__swap_chain_image_views):
            framebufferInfo = VkFramebufferCreateInfo(
                renderPass=self.__render_pass,
                pAttachments=[iv, self.__depth_image_view],
                width=self.__swap_chain_extent.width,
                height=self.__swap_chain_extent.height,
                layers=1,
            )

            self.__swap_chain_framebuffers.append(
                vkCreateFramebuffer(self._logical_device, framebufferInfo, None)
            )

    def __create_command_pool(self):
        queueFamilyIndices = self.__findQueueFamilies(self.__physical_device)

        createInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=queueFamilyIndices.graphicsFamily
        )

        self.__command_pool = vkCreateCommandPool(
            self._logical_device, createInfo, None
        )

    def __create_depth_resources(self):
        depthFormat = self.depthFormat

        self.__depth_image, self.__depth_image_memory = self.__createImage(
            self.__swap_chain_extent.width,
            self.__swap_chain_extent.height,
            depthFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.__depth_image_view = self.__createImageView(
            self.__depth_image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT
        )

        self.__transitionImageLayout(
            self.__depth_image,
            depthFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

    def __findSupportedFormat(self, candidates, tiling, feature):
        for i in candidates:
            props = vkGetPhysicalDeviceFormatProperties(self.__physical_device, i)

            if tiling == VK_IMAGE_TILING_LINEAR and (
                    props.linearTilingFeatures & feature == feature
            ):
                return i
            elif tiling == VK_IMAGE_TILING_OPTIMAL and (
                    props.optimalTilingFeatures & feature == feature
            ):
                return i
        return -1

    @property
    def depthFormat(self):
        return self.__findSupportedFormat(
            [
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D24_UNORM_S8_UINT,
            ],
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        )

    def hasStencilComponent(self, fm):
        return fm == VK_FORMAT_D32_SFLOAT_S8_UINT or fm == VK_FORMAT_D24_UNORM_S8_UINT

    def __createImageView(self, image, imFormat, aspectFlage, dimensions=VK_IMAGE_VIEW_TYPE_2D):
        ssr = VkImageSubresourceRange(
            aspectMask=aspectFlage,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )

        viewInfo = VkImageViewCreateInfo(
            image=image,
            viewType=dimensions,
            format=imFormat,
            subresourceRange=ssr,
        )

        return vkCreateImageView(self._logical_device, viewInfo, None)

    def create_image_3d(self, widht, height, depth, imFormat, tiling, usage, properties):
        imageInfo = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_3D,
            extent=[widht, height, depth],
            mipLevels=1,
            arrayLayers=1,
            format=imFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=tiling,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )

        image = vkCreateImage(self._logical_device, imageInfo, None)

        memReuirements = vkGetImageMemoryRequirements(self._logical_device, image)
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memReuirements.size,
            memoryTypeIndex=self.__findMemoryType(
                memReuirements.memoryTypeBits, properties
            ),
        )

        imageMemory = vkAllocateMemory(self._logical_device, allocInfo, None)

        vkBindImageMemory(self._logical_device, image, imageMemory, 0)

        return (image, imageMemory)

    def __createImage(self, widht, height, imFormat, tiling, usage, properties):
        imageInfo = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D,
            extent=[widht, height, 1],
            mipLevels=1,
            arrayLayers=1,
            format=imFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=tiling,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )

        image = vkCreateImage(self._logical_device, imageInfo, None)

        memReuirements = vkGetImageMemoryRequirements(self._logical_device, image)
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memReuirements.size,
            memoryTypeIndex=self.__findMemoryType(
                memReuirements.memoryTypeBits, properties
            ),
        )

        imageMemory = vkAllocateMemory(self._logical_device, allocInfo, None)

        vkBindImageMemory(self._logical_device, image, imageMemory, 0)

        return (image, imageMemory)

    def __transitionImageLayout(self, image, imFormat, oldLayout, newLayout):
        cmdBuffer = self.__beginSingleTimeCommands()

        subresourceRange = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )
        if newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT
            if self.hasStencilComponent(imFormat):
                subresourceRange.aspectMask = (
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                )

        barrier = VkImageMemoryBarrier(
            oldLayout=oldLayout,
            newLayout=newLayout,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=subresourceRange,
        )

        sourceStage = 0
        destinationStage = 0

        if (
                oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                and newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        ):
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT
        elif (
                oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                and newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        ):
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        elif (
                oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                and newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        ):
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = (
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            )

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
        else:
            raise Exception("unsupported layout transition!")

        vkCmdPipelineBarrier(
            cmdBuffer, sourceStage, destinationStage, 0, 0, None, 0, None, 1, barrier
        )

        self.__endSingleTimeCommands(cmdBuffer)

    def __copyBufferToImage(self, buffer, image, width, height, depth=1):
        cmdbuffer = self.__beginSingleTimeCommands()

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

        self.__endSingleTimeCommands(cmdbuffer)

    def _create_descriptors(self):
        poolInfo = VkDescriptorPoolCreateInfo(
            pPoolSizes=self.descriptor_pools, maxSets=1
        )

        self.__descriptor_pool = vkCreateDescriptorPool(
            self._logical_device, poolInfo, None
        )

        layouts = [self._descriptor_set_layout]
        allocInfo = VkDescriptorSetAllocateInfo(
            descriptorPool=self.__descriptor_pool, pSetLayouts=layouts
        )
        self.__descriptor_set = vkAllocateDescriptorSets(
            self._logical_device, allocInfo
        )

        for w in self.descriptor_writers:
            self.descriptors.append(w(self.__descriptor_set[0]))

        vkUpdateDescriptorSets(
            device=self._logical_device,
            descriptorWriteCount=len(self.descriptors),
            pDescriptorWrites=self.descriptors,
            descriptorCopyCount=0,
            pDescriptorCopies=None
        )

    def __createBuffer(self, size, usage, properties):
        buffer = None
        bufferMemory = None

        bufferInfo = VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = vkCreateBuffer(self._logical_device, bufferInfo, None)

        memRequirements = vkGetBufferMemoryRequirements(self._logical_device, buffer)
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=memRequirements.size,
            memoryTypeIndex=self.__findMemoryType(
                memRequirements.memoryTypeBits, properties
            ),
        )
        bufferMemory = vkAllocateMemory(self._logical_device, allocInfo, None)

        vkBindBufferMemory(self._logical_device, buffer, bufferMemory, 0)

        return (buffer, bufferMemory)

    def __beginSingleTimeCommands(self):
        allocInfo = VkCommandBufferAllocateInfo(
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.__command_pool,
            commandBufferCount=1,
        )
        try:
            commandBuffer = vkAllocateCommandBuffers(self._logical_device, allocInfo)[0]
        except e:
            print(e)
        beginInfo = VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vkBeginCommandBuffer(commandBuffer, beginInfo)

        return commandBuffer

    def __endSingleTimeCommands(self, commandBuffer):
        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(pCommandBuffers=[commandBuffer])

        vkQueueSubmit(self.__graphic_queue, 1, [submitInfo], VK_NULL_HANDLE)
        vkQueueWaitIdle(self.__graphic_queue)

        vkFreeCommandBuffers(
            self._logical_device, self.__command_pool, 1, [commandBuffer]
        )

    def __copyBuffer(self, src, dst, bufferSize):
        commandBuffer = self.__beginSingleTimeCommands()

        # copyRegion = VkBufferCopy(size=bufferSize)
        copyRegion = VkBufferCopy(0, 0, bufferSize)
        vkCmdCopyBuffer(commandBuffer, src, dst, 1, [copyRegion])

        self.__endSingleTimeCommands(commandBuffer)

    def __findMemoryType(self, typeFilter, properties):
        memProperties = vkGetPhysicalDeviceMemoryProperties(self.__physical_device)

        for i, prop in enumerate(memProperties.memoryTypes):
            if (typeFilter & (1 << i)) and (
                    (prop.propertyFlags & properties) == properties
            ):
                return i

        return -1

    def __create_command_buffers(self):
        self.__command_buffers = []

        allocInfo = VkCommandBufferAllocateInfo(
            commandPool=self.__command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__swap_chain_framebuffers),
        )

        self.__command_buffers = vkAllocateCommandBuffers(
            self._logical_device, allocInfo
        )

        for i, buffer in enumerate(self.__command_buffers):
            beginInfo = VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
            )
            vkBeginCommandBuffer(buffer, beginInfo)

            renderArea = VkRect2D([0, 0], self.__swap_chain_extent)
            clearColor = [
                VkClearValue(color=[[0.0, 0.0, 0.0, 1.0]]),
                VkClearValue(depthStencil=[1.0, 0]),
            ]
            renderPassInfo = VkRenderPassBeginInfo(
                renderPass=self.__render_pass,
                framebuffer=self.__swap_chain_framebuffers[i],
                renderArea=renderArea,
                pClearValues=clearColor,
            )

            vkCmdBeginRenderPass(buffer, renderPassInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.__pipeline)

            # todo: implement when you want triangles
            # vkCmdBindVertexBuffers(buffer, 0, 1, [self.__vertex_buffer], [0])
            #   vkCmdBindIndexBuffer(buffer, self.__index_buffer, 0, VK_INDEX_TYPE_UINT32)

            vkCmdBindDescriptorSets(
                buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.__pipeline_layout,
                0,
                1,
                self.__descriptor_set,
                0,
                None,
            )

            # todo: implement when you want triangles
            #   vkCmdDrawIndexed(buffer, len(self.__indices), 1, 0, 0, 0)
            vkCmdDraw(buffer, 3, 1, 0, 0)

            vkCmdEndRenderPass(buffer)

            vkEndCommandBuffer(buffer)

    def __create_semaphores(self):
        semaphore_info = VkSemaphoreCreateInfo()

        self.__image_available_semaphore = vkCreateSemaphore(
            self._logical_device, semaphore_info, None
        )
        self.__render_finished_semaphore = vkCreateSemaphore(
            self._logical_device, semaphore_info, None
        )

    @abc.abstractmethod
    def update_uniform_buffers(self):
        raise NotImplementedError()

    def drawFrame(self):
        #if not self.isExposed():
        #    return

        try:
            imageIndex = vkAcquireNextImageKHR(
                self._logical_device,
                self.__swap_chain,
                18446744073709551615,
                self.__image_available_semaphore,
                VK_NULL_HANDLE,
            )
        except VkErrorSurfaceLostKhr:
            self.__recreate_swap_chain()
            return
        # else:
        #     raise Exception('faild to acquire next image.')

        waitSemaphores = [self.__image_available_semaphore]
        signalSemaphores = [self.__render_finished_semaphore]
        waitStages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        submit = VkSubmitInfo(
            pWaitSemaphores=waitSemaphores,
            pWaitDstStageMask=waitStages,
            pCommandBuffers=[self.__command_buffers[imageIndex]],
            pSignalSemaphores=signalSemaphores,
        )

        vkQueueSubmit(self.__graphic_queue, 1, submit, VK_NULL_HANDLE)

        presenInfo = VkPresentInfoKHR(
            pWaitSemaphores=signalSemaphores,
            pSwapchains=[self.__swap_chain],
            pImageIndices=[imageIndex],
        )

        try:
            vkQueuePresentKHR(self.__present_queue, presenInfo)
        except VkErrorOutOfDateKhr:
            self.__recreate_swap_chain()

        if self.enable_validation_layers:
            vkQueueWaitIdle(self.__present_queue)

    def _create_shader_module(self, shaderFile):
        with open(shaderFile, "rb") as sf:
            code = sf.read()

            createInfo = VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)

            return vkCreateShaderModule(self._logical_device, createInfo, None)

    def __chooseSwapSurfaceFormat(self, formats):
        if len(formats) == 1 and formats[0].format == VK_FORMAT_UNDEFINED:
            return [VK_FORMAT_B8G8R8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]

        for i in formats:
            if (
                    i.format == VK_FORMAT_B8G8R8_UNORM
                    and i.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                return i

        return formats[0]

    def __chooseSwapPresentMode(self, presentModes):
        bestMode = VK_PRESENT_MODE_FIFO_KHR

        for i in presentModes:
            if i == VK_PRESENT_MODE_FIFO_KHR:
                return i
            elif i == VK_PRESENT_MODE_MAILBOX_KHR:
                return i
            elif i == VK_PRESENT_MODE_IMMEDIATE_KHR:
                return i

        return bestMode

    def __chooseSwapExtent(self, capabilities):
        w, h = self.get_window_size()
        width = max(
            capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, w)
        )
        height = max(
            capabilities.minImageExtent.height,
            min(capabilities.maxImageExtent.height, h),
        )
        return VkExtent2D(width, height)

    def __querySwapChainSupport(self, device):
        detail = SwapChainSupportDetails()

        detail.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            device, self.__surface
        )
        detail.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, self.__surface)
        detail.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(
            device, self.__surface
        )
        return detail

    def __isDeviceSuitable(self, device):
        indices = self.__findQueueFamilies(device)

        extensionsSupported = self.__checkDeviceExtensionSupport(device)

        swapChainAdequate = False
        if extensionsSupported:
            swapChainSupport = self.__querySwapChainSupport(device)
            swapChainAdequate = (swapChainSupport.formats is not None) and (
                    swapChainSupport.presentModes is not None
            )

        supportedFeatures = vkGetPhysicalDeviceFeatures(device)

        return (
                indices.isComplete
                and extensionsSupported
                and swapChainAdequate
                and supportedFeatures.samplerAnisotropy
        )

    def __checkDeviceExtensionSupport(self, device):
        availableExtensions = vkEnumerateDeviceExtensionProperties(device, None)

        aen = [i.extensionName for i in availableExtensions]
        for i in self.device_extensions:
            if i not in aen:
                return False

        return True

    def __findQueueFamilies(self, device):
        indices = QueueFamilyIndices()

        familyProperties = vkGetPhysicalDeviceQueueFamilyProperties(device)
        for i, prop in enumerate(familyProperties):
            if prop.queueCount > 0 and prop.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                indices.graphicsFamily = i

            presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(
                device, i, self.__surface
            )

            if prop.queueCount > 0 and presentSupport:
                indices.presentFamily = i

            if indices.isComplete:
                break

        return indices

    def __getRequiredExtensions(self):
        extensions = [
            e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)
        ]

        if self.enable_validation_layers:
            extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        if self.wm_info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
            extensions.append("VK_KHR_win32_surface")
        elif self.wm_info.subsystem == sdl2.SDL_SYSWM_X11:
            extensions.append("VK_KHR_xlib_surface")
        elif self.wm_info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
            extensions.append("VK_KHR_wayland_surface")
        else:
            raise Exception("Platform not supported")

        return extensions

    @abc.abstractmethod
    def create_uniforms(self):
        raise NotImplementedError()

    def create_3d_texture(self, npz_file, binding, vkformat=VK_FORMAT_R8_SINT):
        _image = np.load(npz_file)
        _image = _image[_image.files[0]]
        shape = _image.shape
        width = _image.shape[0]
        height = _image.shape[1]
        depth = _image.shape[2]
        if vkformat == VK_FORMAT_R8G8B8A8_UNORM:
            _image = _image.astype(np.uint8)
            image_size = width * height * depth * 4
        elif vkformat == VK_FORMAT_R8_SINT:
            _image = _image.astype(np.uint8)
            image_size = width * height * depth
        else:
            raise NotImplementedError(f"pls fix code for vkformat: {(vkformat)} (and search for the right name)")

        staging_buffer, staging_mem = self.__createBuffer(
            image_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        data = vkMapMemory(self._logical_device, staging_mem, 0, image_size, 0)
        ffi.memmove(data, _image.tobytes(), image_size)
        vkUnmapMemory(self._logical_device, staging_mem)

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

        self.__transitionImageLayout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        )
        self.__copyBufferToImage(staging_buffer, texture_image, width, height, depth)
        self.__transitionImageLayout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )

        vkDestroyBuffer(self._logical_device, staging_buffer, None)
        vkFreeMemory(self._logical_device, staging_mem, None)

        self.__texture_image_views.append(self.__createImageView(
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
            self._logical_device, sampler_info, None
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

    def create_2d_texture(self, file, binding, vkformat=VK_FORMAT_R8G8B8A8_UNORM):
        _image = Image.open(file)
        _image.putalpha(1)
        width = _image.width
        height = _image.height
        if vkformat == VK_FORMAT_R8G8B8A8_UNORM:
            imageSize = width * height * 4
            shape = [width, height, 4]
        else:
            raise NotImplementedError(f"pls fix code for vkformat: {(vkformat)} (and search for the right name)")

        stagingBuffer, stagingMem = self.__createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        data = vkMapMemory(self._logical_device, stagingMem, 0, imageSize, 0)
        ffi.memmove(data, _image.tobytes(), imageSize)
        vkUnmapMemory(self._logical_device, stagingMem)

        del _image

        texture_image, texture_image_memory = self.__createImage(
            width,
            height,
            vkformat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        self.__texture_images.append(texture_image)
        self.__texture_image_memories.append(texture_image_memory)

        self.__transitionImageLayout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        )
        self.__copyBufferToImage(stagingBuffer, texture_image, width, height)
        self.__transitionImageLayout(
            texture_image,
            vkformat,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )

        vkDestroyBuffer(self._logical_device, stagingBuffer, None)
        vkFreeMemory(self._logical_device, stagingMem, None)

        self.__texture_image_views.append(self.__createImageView(
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
            self._logical_device, sampler_info, None
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

    def create_ubo(self, ubo, binding):
        __uniform_buffer, __uniform_buffer_memory = self.__createBuffer(
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
            type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=ubo.nbytes
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
        self.drawFrame()

    def sdl2_event_check(self):
        for event in sdl2.ext.get_events():
            if event.type in self.event_dict:
                self.event_dict[event.type](event)

    def loop_once(self):
        self.sdl2_event_check()
        self.render()

    def resize_event(self, event):
        if event.size() != event.oldSize():
            self.__recreate_swap_chain()


# todo: unused. Implement whenever I want to use meshes again.
'''
    def _create_vertex_buffer(self, vertices):
        bufferSize = vertices.nbytes

        stagingBuffer, stagingMemory = self.__createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        data = vkMapMemory(self._logical_device, stagingMemory, 0, bufferSize, 0)
        vertePtr = ffi.cast("float *", vertices.ctypes.data)
        ffi.memmove(data, vertePtr, bufferSize)
        vkUnmapMemory(self._logical_device, stagingMemory)

        vertex_buffer, vertex_buffer_memory = self.__createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.__copyBuffer(stagingBuffer, vertex_buffer, bufferSize)

        vkDestroyBuffer(self._logical_device, stagingBuffer, None)
        vkFreeMemory(self._logical_device, stagingMemory, None)

        return vertex_buffer, vertex_buffer_memory

    def _create_index_buffer(self, indices):
        bufferSize = indices.nbytes

        stagingBuffer, stagingMemory = self.__createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        data = vkMapMemory(self._logical_device, stagingMemory, 0, bufferSize, 0)

        indicesPtr = ffi.cast("uint16_t*", self.__indices.ctypes.data)
        ffi.memmove(data, indicesPtr, bufferSize)

        vkUnmapMemory(self._logical_device, stagingMemory)

        self.__index_buffer, self.__index_buffer_memory = self.__createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        self.__copyBuffer(stagingBuffer, self.__index_buffer, bufferSize)

        vkDestroyBuffer(self._logical_device, stagingBuffer, None)
        vkFreeMemory(self._logical_device, stagingMemory, None)
'''


class AbstractUBO(object):
    @abc.abstractmethod
    def to_bytes(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def nbytes(self):
        raise NotImplementedError()
