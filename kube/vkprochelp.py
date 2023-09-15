import numpy as np
from vulkan import VK_ERROR_EXTENSION_NOT_PRESENT, vkGetInstanceProcAddr, vkGetDeviceProcAddr, \
    VkVertexInputBindingDescription, VK_VERTEX_INPUT_RATE_VERTEX, VkVertexInputAttributeDescription, \
    VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32_SFLOAT


class InstanceProcAddr(object):
    """Dynamically check for and load function from Vulkan"""

    T = None

    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        func_name = self.__func.__name__
        func = InstanceProcAddr.procfunc(func_name)
        if func:
            return func(*args, **kwargs)
        else:
            return VK_ERROR_EXTENSION_NOT_PRESENT

    @staticmethod
    def procfunc(func_name):
        return vkGetInstanceProcAddr(InstanceProcAddr.T, func_name)


class DeviceProcAddr(InstanceProcAddr):
    """Gets function addresses specific to devices rather than instances."""

    @staticmethod
    def procfunc(func_name):
        return vkGetDeviceProcAddr(InstanceProcAddr.T, func_name)


@InstanceProcAddr
def vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):  # NOSONAR
    """Create a debug report callback object"""
    pass


@InstanceProcAddr
def vkDestroyDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):  # NOSONAR
    """Destroy a debug report callback object"""
    pass


@InstanceProcAddr
def vkCreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator):  # NOSONAR
    """Create a VkSurfaceKHR object for an Win32 native window"""
    pass


@InstanceProcAddr
def vkDestroySurfaceKHR(instance, surface, pAllocator):  # NOSONAR
    """Destroy a VkSurfaceKHR object"""
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface):  # NOSONAR
    """Query if presentation is supported"""
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface):  # NOSONAR
    """Query surface capabilities"""
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface):  # NOSONAR
    """Query color formats supported by surface"""
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface):  # NOSONAR
    """Query supported presentation modes"""
    pass


@DeviceProcAddr
def vkCreateSwapchainKHR(device, pCreateInfo, pAllocator):  # NOSONAR
    """Create a swapchain"""
    pass


@DeviceProcAddr
def vkDestroySwapchainKHR(device, swapchain, pAllocator):  # NOSONAR
    """Destroy a swapchain object"""
    pass


@DeviceProcAddr
def vkGetSwapchainImagesKHR(device, swapchain):  # NOSONAR
    """Obtain the array of presentable images associated with a swapchain"""
    pass


@DeviceProcAddr
def vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence):  # NOSONAR
    """Retrieve the index of the next available presentable image"""
    pass


@DeviceProcAddr
def vkQueuePresentKHR(queue, pPresentInfo):  # NOSONAR
    """Queue an image for presentation"""
    pass


class Win32misc(object):
    @staticmethod
    def get_instance(h_wnd):
        from cffi import FFI as _FFI

        _ffi = _FFI()
        _ffi.cdef("long __stdcall GetWindowLongA(void* hWnd, int nIndex);")
        _lib = _ffi.dlopen("User32.dll")
        return _lib.GetWindowLongA(_ffi.cast("void*", h_wnd), -6)  # GWL_HINSTANCE


class SwapChainSupportDetails(object):
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.present_modes = None


class Vertex(object):
    POS = np.array([0, 0, 0], np.float32)
    COLOR = np.array([0, 0, 0], np.float32)
    TEXCOORD = np.array([0, 0], np.float32)

    @staticmethod
    def get_binding_description():
        binding_description = VkVertexInputBindingDescription(
            binding=0,
            stride=Vertex.POS.nbytes + Vertex.COLOR.nbytes + Vertex.TEXCOORD.nbytes,
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX,
        )

        return binding_description

    @staticmethod
    def get_attribute_descriptions():
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