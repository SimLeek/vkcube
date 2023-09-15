from vulkan import vkGetPhysicalDeviceQueueFamilyProperties, VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_COMPUTE_BIT

from kube.vkprochelp import SwapChainSupportDetails, vkGetPhysicalDeviceSurfaceCapabilitiesKHR, \
    vkGetPhysicalDeviceSurfaceFormatsKHR, vkGetPhysicalDeviceSurfacePresentModesKHR, \
    vkGetPhysicalDeviceSurfaceSupportKHR


class QueueFamilyIndices(object):
    def __init__(self):
        self.graphics_family = -1
        self.present_family = -1
        self.compute_family = -1

    @property
    def is_complete(self):
        return self.graphics_family >= 0 and self.present_family >= 0 and self.compute_family >= 0


def query_swap_chain_support(device, surface):
    detail = SwapChainSupportDetails()

    detail.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        device, surface
    )
    detail.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface)
    detail.present_modes = vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface
    )
    return detail


def find_queue_families(device, surface):
    indices = QueueFamilyIndices()

    family_properties = vkGetPhysicalDeviceQueueFamilyProperties(device)
    for i, prop in enumerate(family_properties):
        if prop.queueCount > 0 and prop.queueFlags & VK_QUEUE_GRAPHICS_BIT:
            indices.graphics_family = i

        if prop.queueCount > 0 and prop.queueFlags & VK_QUEUE_COMPUTE_BIT:
            indices.compute_family = i

        present_support = vkGetPhysicalDeviceSurfaceSupportKHR(
            device, i, surface
        )

        if prop.queueCount > 0 and present_support:
            indices.present_family = i

        if indices.is_complete:
            break

    return indices
