from vulkan import VkDescriptorPoolCreateInfo, vkCreateDescriptorPool, VkDescriptorSetAllocateInfo, \
    vkAllocateDescriptorSets, vkUpdateDescriptorSets

from kube.vulkan_device import VulkanDevice
from kube.vulkan_pipelines import VulkanPipelines


class VulkanDescriptors(object):
    def __init__(self, vulkan_device:VulkanDevice, vulkan_pipelines:VulkanPipelines):
        self.vulkan_device = vulkan_device
        self.vulkan_pipelines = vulkan_pipelines
        self.descriptor_set = None
        self.descriptors = []

    def setup(self, descriptor_pools, descriptor_writers):
        self._create_descriptors(descriptor_pools, descriptor_writers)

    def _create_descriptors(self, descriptor_pools, descriptor_writers):
        pool_info = VkDescriptorPoolCreateInfo(
            pPoolSizes=descriptor_pools, maxSets=1
        )

        self.__descriptor_pool = vkCreateDescriptorPool(
            self.vulkan_device.logical_device, pool_info, None
        )

        layouts = [self.vulkan_pipelines.descriptor_set_layout]
        alloc_info = VkDescriptorSetAllocateInfo(
            descriptorPool=self.__descriptor_pool, pSetLayouts=layouts
        )
        self.descriptor_set = vkAllocateDescriptorSets(
            self.vulkan_device.logical_device, alloc_info
        )

        for w in descriptor_writers:
            self.descriptors.append(w(self.descriptor_set[0]))

        vkUpdateDescriptorSets(
            device=self.vulkan_device.logical_device,
            descriptorWriteCount=len(self.descriptors),
            pDescriptorWrites=self.descriptors,
            descriptorCopyCount=0,
            pDescriptorCopies=None
        )