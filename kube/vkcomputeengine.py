import os
import kp
import numpy as np

def compile_file(file):
    f = open(file)
    f_dir = os.path.dirname(os.path.abspath(f.name))
    f_name = os.path.basename(f.name)

    cmd = f"{os.environ['VULKAN_SDK']}{os.sep}Bin{os.sep}glslc.exe {f_dir}{os.sep}{f_name} -o {f_dir}{os.sep}{f_name}.spv"
    print(cmd)
    os.system( cmd )
    return open(f"{f_name}.spv", "rb").read()

def kompute(shader):
    # 1. Create Kompute Manager with default settings (device 0, first queue and no extensions)
    mgr = kp.Manager()

    # minecraft block is 16x16x256, with cube sizes of 1 meter
    # our cube sizes are 1/32 of a meter, about 1 inch, so for the same 16x16 size, we need 16-32=512
    cube_size = 512
    # however, this will result in almost a gigabyte of data... so it needs to be compressed

    tensor_in_a = mgr.tensor_t(np.asarray([cube_size,cube_size,cube_size], dtype=np.int32))
    tensor_in_b = mgr.tensor(np.asarray([0.02,0.0,0.0], dtype=np.float32))
    # 2. Create and initialise Kompute Tensors through manager
    tensor_out_a = mgr.tensor_t(np.zeros([cube_size,cube_size,cube_size], dtype=np.float32))

    params = [tensor_in_a, tensor_in_b, tensor_out_a]

    # 3. Create algorithm based on shader (supports buffers & push/spec constants)
    workgroup = [cube_size,cube_size,cube_size]
    push_consts_a = []

    # See documentation shader section for compile_source
    spirv = compile_file(shader)

    algo = mgr.algorithm(params, spirv, workgroup, [], [])

    # 4. Run operation synchronously using sequence
    (mgr.sequence()
        .record(kp.OpTensorSyncDevice(params))
        .record(kp.OpAlgoDispatch(algo)) # Binds default push consts provided
        .eval()) # evaluates the two recorded ops
        #.record(kp.OpAlgoDispatch(algo, push_consts_a)) # Overrides push consts
        #.eval()) # evaluates only the last recorded op

    # 5. Sync results from the GPU asynchronously
    sq = mgr.sequence()
    sq.eval_async(kp.OpTensorSyncLocal(params))

    # ... Do other work asynchronously whilst GPU finishes
    sq.eval_await()

    # Prints the first output which is: { 4, 8, 12 }
    print(tensor_out_a.data())

if __name__ == "__main__":
    kompute('compute3DNoiseLayer.comp')