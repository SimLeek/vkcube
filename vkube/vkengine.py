# -*- coding: UTF-8 -*-
import vkube.sdl_include
import sdl2
import sdl2.ext
from vulkan import *
import time
from vkube.vkabstractapplication import VKAbstractApplication, AbstractUBO
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


class VKOctreeApplication(VKAbstractApplication):
    def __init__(self):
        self.descriptors = []

        self.input_texture_infos_ubo = InputTextureInfosUBO()
        self.user_input_ubo = UserInputUBO()
        self.capturing_mouse = True

        super().__init__()

        self.event_dict[sdl2.SDL_KEYDOWN] = self.key_down_event

        # self.event_dict[sdl2.SDL_WINDOWEVENT_RESIZED] = self.resize_event

    # def resize_event(self, event):
    #    super().resize_event()

    def create_descriptor_set_layout(self):
        # matches all the bound inputs to each shader

        input_texture_info_ubo_binding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
        )

        i_channel_0_sampler_binding = VkDescriptorSetLayoutBinding(
            binding=1,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        )

        i_channel_1_sampler_binding = VkDescriptorSetLayoutBinding(
            binding=2,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
        )

        i_channel_2_sampler_binding = VkDescriptorSetLayoutBinding(
            binding=3,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
        )

        i_channel_3_sampler_binding = VkDescriptorSetLayoutBinding(
            binding=4,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
        )

        user_input_ubo_binding = VkDescriptorSetLayoutBinding(
            binding=5,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,
        )

        layout_info = VkDescriptorSetLayoutCreateInfo(
            pBindings=[input_texture_info_ubo_binding,
                       i_channel_0_sampler_binding,
                       i_channel_1_sampler_binding,
                       i_channel_2_sampler_binding,
                       i_channel_3_sampler_binding,
                       user_input_ubo_binding]
        )

        self._descriptor_set_layout = vkCreateDescriptorSetLayout(
            self._logical_device, layout_info, None
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

    def update_uniform_buffers(self):
        current_time = time.time()

        t = current_time - self._start_time

        x, y, s = self.get_mouse()
        w, h = self.get_window_size()

        self.user_input_ubo.iTime[0] = t
        self.user_input_ubo.iMouse[:3] = x, y, s

        self.input_texture_infos_ubo.i_resolution[:] = w, h

        ui_struct = self.user_input_ubo.to_bytes()
        res_struct = self.input_texture_infos_ubo.to_bytes()

        data1 = vkMapMemory(
            self._logical_device, self._uniform_buffer_memories[1], 0, len(ui_struct), 0
        )
        data2 = vkMapMemory(
            self._logical_device, self._uniform_buffer_memories[0], 0, len(res_struct), 0
        )

        ffi.memmove(data1, ui_struct, len(ui_struct))
        ffi.memmove(data2, res_struct, len(res_struct))

        vkUnmapMemory(self._logical_device, self._uniform_buffer_memories[0])
        vkUnmapMemory(self._logical_device, self._uniform_buffer_memories[1])

    def key_down_event(self, e:sdl2.SDL_Event):
        """Events for one time key presses, like pause"""
        if e.key.keysym.scancode == sdl2.SDL_SCANCODE_ESCAPE:
            self.capturing_mouse = not self.capturing_mouse


    def sdl2_event_check(self):

        if self.capturing_mouse:
            x, y, s = self.get_mouse()
            w, h = self.get_window_size()
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

                sdl2.SDL_WarpMouseInWindow(self.window, w // 2, h // 2)

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
            if event.type in self.event_dict:
                self.event_dict[event.type](event)




if __name__ == "__main__":
    app = VKOctreeApplication()
    while app.alive:
        app.loop_once()
