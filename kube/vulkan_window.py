import ctypes

import sdl2


class VulkanWindow(object):
    """SDL2 Window for Vulkan"""

    def __init__(self, parent_obj, width=1280, height=720):
        self.parent = parent_obj

        self.width = width
        self.height = height

        self.window, self.wm_info = self.init_sdl_window()
        sdl2.SDL_WarpMouseInWindow(self.window, width // 2, height // 2)
        sdl2.SDL_SetWindowTitle(self.window, ctypes.c_char_p(b"VKCube"))

        self.event_dict = dict()
        self.event_dict[sdl2.SDL_WINDOWEVENT_RESIZED] = self.resize_event
        self.event_dict[sdl2.SDL_QUIT] = self.unalive_self

    def unalive_self(self, e):
        self.alive = False

    def resize_event(self, event):
        if event.size() != event.oldSize():
            self.parent.recreate_swap_chain()

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

    def sdl2_event_check(self):
        x, y, s = self.get_mouse()
        w, h = self.get_window_size()
        # print(x, y)

        if x != w // 2 or y != h // 2:
            diff_x = x - w // 2
            diff_y = y - h // 2
            print(diff_x, diff_y)
            sdl2.SDL_WarpMouseInWindow(self.window, w // 2, h // 2)

        for event in sdl2.ext.get_events():
            if event.type in self.event_dict:
                self.event_dict[event.type](event)