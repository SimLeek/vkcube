# -*- coding: UTF-8 -*-
import kube.sdl_include
import sdl2
import sdl2.ext
from vulkan import *
import time
from kube.vkabstractcomputer import VKAbstractComputer, AbstractUBO
import numpy as np
import struct

class InputTextureInfosUBO(AbstractUBO):
    def __init__(self):
        self.iChannelResolution = np.zeros((4, 3), np.int32)
        self.i_resolution = np.zeros((2,), np.int32)

    def to_bytes(self):
        return struct.pack("<" + f"iii{'x' * 4}" * 4 + f"ii",
                                 *self.iChannelResolution[0],
                                 *self.iChannelResolution[1],
                                 *self.iChannelResolution[2],
                                 *self.iChannelResolution[3],
                                 *self.i_resolution)

    @property
    def nbytes(self):
        return len(self.to_bytes())


class UserInputUBO(AbstractUBO):
    def __init__(self):
        self.iTime = np.zeros((4,), np.float32)
        self.iMouse = np.zeros((4,), np.float32)
        self.ipos = np.zeros((3,), np.float32)
        self.iRotation = np.zeros((2,), np.float32)

    def to_bytes(self):
        return struct.pack(f"<f{'x' * 12}fff{'x' * 4}fff{'x' * 4}ff",
                           self.iTime[0],
                           *self.iMouse[:3], *self.ipos, *self.iRotation)

    @property
    def nbytes(self):
        return len(self.to_bytes())


class VKOctreeApplication(VKAbstractComputer):
    def __init__(self):
        self.input_texture_infos_ubo = InputTextureInfosUBO()
        self.user_input_ubo = UserInputUBO()
        self.capturing_mouse = True

        super().__init__()

        self.vulkan_window.event_dict[sdl2.SDL_KEYDOWN] = self.key_down_event

        # self.event_dict[sdl2.SDL_WINDOWEVENT_RESIZED] = self.resize_event

    # def resize_event(self, event):
    #    super().resize_event()

    def create_descriptor_set_layout(self):
        # matches all the bound inputs to each shader

        input_texture_info_ubo_binding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        )

        i_channel_0_sampler_binding = VkDescriptorSetLayoutBinding(
            binding=1,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        )

        layout_info = VkDescriptorSetLayoutCreateInfo(
            pBindings=[input_texture_info_ubo_binding,
                       i_channel_0_sampler_binding]
        )

        self.vulkan_pipelines.descriptor_set_layout = vkCreateDescriptorSetLayout(
            self.vulkan_device.logical_device, layout_info, None
        )

    def create_uniforms(self):
        self.create_ubo(self.input_texture_infos_ubo, 0)
        shape1 = self.create_3d_texture('clouds.npz', 1, VK_FORMAT_R8_UNORM)
        self.input_texture_infos_ubo.iChannelResolution[0, :3] = shape1[:3]
        shape2 = self.create_3d_texture('cloud_rgba.npz', 2, VK_FORMAT_R8G8B8A8_UNORM)
        self.input_texture_infos_ubo.iChannelResolution[1, :3] = shape2[:3]
        shape3 = self.create_2d_texture('albedo.png', 3, VK_FORMAT_R8G8B8A8_UNORM)
        self.input_texture_infos_ubo.iChannelResolution[2, :3] = shape3[:3]
        shape4 = self.create_2d_texture('tk.png', 4, VK_FORMAT_R8G8B8A8_UNORM)
        self.input_texture_infos_ubo.iChannelResolution[3, :3] = shape4[:3]
        self.create_ubo(self.user_input_ubo, 5)

    def create_shaders(self):
        vertex_shader_mode = self._create_shader_module('skip.vert.spv')

        vertex_shader_stage_info = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_VERTEX_BIT, module=vertex_shader_mode, pName="main"
        )

        fragment_shader_mode = self._create_shader_module('sdf.frag.spv')

        fragment_shader_stage_info = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_FRAGMENT_BIT, module=fragment_shader_mode, pName="main"
        )

        return [vertex_shader_stage_info, fragment_shader_stage_info]

    def create_compute_shader(self):
        compute_shader_module = self._create_shader_module('compute3DNoiseLayer.comp.spv')

        compute_shader_stage_info = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=compute_shader_module,
            pName="main",  # Adjust the entry point name if necessary
        )

        return compute_shader_stage_info
        return None

    def update_uniform_buffers(self):
        current_time = time.time()

        t = current_time - self._start_time

        x, y, s = self.vulkan_window.get_mouse()
        w, h = self.vulkan_window.get_window_size()

        self.user_input_ubo.iTime[0] = t
        self.user_input_ubo.iMouse[:3] = x, y, s

        self.input_texture_infos_ubo.i_resolution[:] = w, h

        ui_struct = self.user_input_ubo.to_bytes()
        res_struct = self.input_texture_infos_ubo.to_bytes()

        data1 = vkMapMemory(
            self.vulkan_device.logical_device, self._uniform_buffer_memories[1], 0, len(ui_struct), 0
        )
        data2 = vkMapMemory(
            self.vulkan_device.logical_device, self._uniform_buffer_memories[0], 0, len(res_struct), 0
        )

        ffi.memmove(data1, ui_struct, len(ui_struct))
        ffi.memmove(data2, res_struct, len(res_struct))

        vkUnmapMemory(self.vulkan_device.logical_device, self._uniform_buffer_memories[0])
        vkUnmapMemory(self.vulkan_device.logical_device, self._uniform_buffer_memories[1])

    def key_down_event(self, e:sdl2.SDL_Event):
        """Events for one time key presses, like pause"""
        if e.key.keysym.scancode == sdl2.SDL_SCANCODE_ESCAPE:
            self.capturing_mouse = not self.capturing_mouse


    def sdl2_event_check(self):

        if self.capturing_mouse:
            x, y, s = self.vulkan_window.get_mouse()
            w, h = self.vulkan_window.get_window_size()
            #print(x, y)

            if x!=w//2 or y!=h//2:
                diff_x = x-w//2
                diff_y = y-h//2

                self.user_input_ubo.iRotation[0] -= diff_x /w
                self.user_input_ubo.iRotation[1] -= diff_y /h
                if self.user_input_ubo.iRotation[1] > np.pi/2:
                    self.user_input_ubo.iRotation[1] = np.pi/2 -.01
                elif self.user_input_ubo.iRotation[1] < -np.pi/2:
                    self.user_input_ubo.iRotation[1] = -np.pi/2 +.01

                sdl2.SDL_WarpMouseInWindow(self.vulkan_window.window, w // 2, h // 2)

        key_states = sdl2.SDL_GetKeyboardState(None)
        if key_states[sdl2.SDL_SCANCODE_W]:
            self.user_input_ubo.ipos[0] -= np.sin(self.user_input_ubo.iRotation[0])*.01
            self.user_input_ubo.ipos[2] -= np.cos(self.user_input_ubo.iRotation[0])*.01
        if key_states[sdl2.SDL_SCANCODE_S]:
            self.user_input_ubo.ipos[0] += np.sin(self.user_input_ubo.iRotation[0]) * .01
            self.user_input_ubo.ipos[2] += np.cos(self.user_input_ubo.iRotation[0]) * .01
        if key_states[sdl2.SDL_SCANCODE_A]:
            self.user_input_ubo.ipos[0] -= np.cos(self.user_input_ubo.iRotation[0]) * .01
            self.user_input_ubo.ipos[2] += np.sin(self.user_input_ubo.iRotation[0]) * .01
        if key_states[sdl2.SDL_SCANCODE_D]:
            self.user_input_ubo.ipos[0] += np.cos(self.user_input_ubo.iRotation[0]) * .01
            self.user_input_ubo.ipos[2] -= np.sin(self.user_input_ubo.iRotation[0]) * .01
        if key_states[sdl2.SDL_SCANCODE_SPACE]:
            self.user_input_ubo.ipos[1] -= .01
        if key_states[sdl2.SDL_SCANCODE_LSHIFT]:
            self.user_input_ubo.ipos[1] += .01
        #elif key_states[sdl2.SDL_SCANCODE_ESCAPE]:
        #    self.capturing_mouse = not self.capturing_mouse

        for event in sdl2.ext.get_events():
            if event.type in self.vulkan_window.event_dict:
                self.vulkan_window.event_dict[event.type](event)

    def write_compute_ubos(self, data):
        data_struct = data.to_bytes()

        # Map the memory of the UBOs
        ui_data = vkMapMemory(
            self.vulkan_device.logical_device, self._uniform_buffer_memories[1], 0, len(ui_struct), 0
        )
        res_data = vkMapMemory(
            self.vulkan_device.logical_device, self._uniform_buffer_memories[0], 0, len(res_struct), 0
        )

        # Copy the data to the UBO memory
        ffi.memmove(ui_data, ui_struct, len(ui_struct))
        ffi.memmove(res_data, res_struct, len(res_struct))

        # Unmap the memory
        vkUnmapMemory(self.vulkan_device.logical_device, self._uniform_buffer_memories[0])
        vkUnmapMemory(self.vulkan_device.logical_device, self._uniform_buffer_memories[1])

        # Synchronize with the compute shader
        vkQueueSubmit(self.vulkan_device.compute_queue, 1, VkSubmitInfo(
            waitSemaphoreCount=0,
            pWaitSemaphores=None,
            pWaitDstStageMask=None,
            commandBufferCount=0,
            pCommandBuffers=None,
            signalSemaphoreCount=1,
            pSignalSemaphores=self.vulkan_semaphores.write_semaphore
        ), VK_NULL_HANDLE)
        vkQueueWaitIdle(self.vulkan_device.compute_queue)  # Wait for compute to finish
        vkResetFences(self.vulkan_device.logical_device, 1, [self.vulkan_semaphores.in_flight_fences[self.current_frame]])
        self.current_frame = (self.current_frame + 1) % self.vulkan_semaphores.MAX_FRAMES_IN_FLIGHT

    def read_compute_ubos(self):
        # Synchronize with the compute shader
        vkWaitForFences(self.vulkan_device.logical_device, 1, [self.vulkan_semaphores.in_flight_fences[self.current_frame]], VK_TRUE, UINT64_MAX)
        vkQueueSubmit(self.vulkan_device.compute_queue, 1, VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=self.vulkan_semaphores.write_semaphore,
            pWaitDstStageMask=None,
            commandBufferCount=1,
            pCommandBuffers=self.__command_buffers[self.current_frame],
            signalSemaphoreCount=1,
            pSignalSemaphores=self.vulkan_semaphores.read_semaphore
        ), VK_NULL_HANDLE)
        vkQueueWaitIdle(self.vulkan_device.compute_queue)  # Wait for compute to finish
        vkResetFences(self.vulkan_device.logical_device, 1, [self.vulkan_semaphores.in_flight_fences[self.current_frame]])
        self.current_frame = (self.current_frame + 1) % self.vulkan_semaphores.MAX_FRAMES_IN_FLIGHT

        # Map the memory of the UBOs
        ui_data = vkMapMemory(
            self.vulkan_device.logical_device, self._uniform_buffer_memories[1], 0, len(UserInputUBO), 0
        )
        res_data = vkMapMemory(
            self.vulkan_device.logical_device, self._uniform_buffer_memories[0], 0, len(InputTextureInfosUBO), 0
        )

        # Create UBO objects from the mapped memory
        ui_struct = ffi.cast("UserInputUBO*", ui_data)
        res_struct = ffi.cast("InputTextureInfosUBO*", res_data)

        # Read data from the UBOs
        # You can access ui_struct and res_struct here

        # Unmap the memory
        vkUnmapMemory(self.vulkan_device.logical_device, self._uniform_buffer_memories[0])
        vkUnmapMemory(self.vulkan_device.logical_device, self._uniform_buffer_memories[1])

        return ui_struct, res_struct




if __name__ == "__main__":
    app = VKOctreeApplication()
    while app.alive:
        app.render_once()
        return_str = app.compute_once("text")
