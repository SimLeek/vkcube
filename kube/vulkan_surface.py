import sdl2
from vulkan import vkGetInstanceProcAddr, VkXlibSurfaceCreateInfoKHR, VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR, \
    VkWaylandSurfaceCreateInfoKHR, VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR, VkWin32SurfaceCreateInfoKHR, \
    VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR

from kube.vkprochelp import vkDestroySurfaceKHR


class VulkanSurface(object):
    def __init__(self, vulkan_instance, vulkan_window):
        self.vulkan_instance = vulkan_instance
        self.vulkan_window = vulkan_window

        self.surface = None

    def __del__(self):
        if self.surface:
            vkDestroySurfaceKHR(self.vulkan_instance.instance, self.surface, None)

    def setup(self):
        def surface_xlib():
            print("Create Xlib surface")
            vk_create_xlib_surface_khr = vkGetInstanceProcAddr(
                self.vulkan_instance.instance, "vkCreateXlibSurfaceKHR"
            )
            surface_create = VkXlibSurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                dpy=self.vulkan_window.wm_info.info.x11.display,
                window=self.vulkan_window.wm_info.info.x11.window,
                flags=0,
            )
            return vk_create_xlib_surface_khr(self.vulkan_instance.instance, surface_create, None)

        def surface_wayland():
            print("Create wayland surface")
            vk_create_wayland_surface_khr = vkGetInstanceProcAddr(
                self.vulkan_instance.instance, "vkCreateWaylandSurfaceKHR"
            )
            surface_create = VkWaylandSurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                display=self.vulkan_window.wm_info.info.wl.display,
                surface=self.vulkan_window.wm_info.info.wl.surface,
                flags=0,
            )
            return vk_create_wayland_surface_khr(self.vulkan_instance.instance, surface_create, None)

        def surface_win32():
            def get_instance(h_wnd):
                """Hack needed before SDL 2.0.6"""
                from cffi import FFI

                _ffi = FFI()
                _ffi.cdef("long __stdcall GetWindowLongA(void* hWnd, int nIndex);")
                _lib = _ffi.dlopen("User32.dll")
                return _lib.GetWindowLongA(_ffi.cast("void*", h_wnd), -6)

            print("Create windows surface")
            vk_create_win32_surface_khr = vkGetInstanceProcAddr(
                self.vulkan_instance.instance, "vkCreateWin32SurfaceKHR"
            )
            surface_create = VkWin32SurfaceCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                hinstance=get_instance(self.vulkan_window.wm_info.info.win.window),
                hwnd=self.vulkan_window.wm_info.info.win.window,
                flags=0,
            )
            return vk_create_win32_surface_khr(self.vulkan_instance.instance, surface_create, None)

        surface_mapping = {
            sdl2.SDL_SYSWM_X11: surface_xlib,
            sdl2.SDL_SYSWM_WAYLAND: surface_wayland,
            sdl2.SDL_SYSWM_WINDOWS: surface_win32,
        }

        self.surface = surface_mapping[self.vulkan_window.wm_info.subsystem]()