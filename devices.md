# NVIDIA CUDA Python编程框架--Warp开发文档--设备
Warp 为系统中所有支持的计算设备分配唯一的字符串别名。 当前有一个 CPU 设备公开为“cpu”。 每个支持 CUDA 的 GPU 都会获得一个“cuda:i”形式的别名，其中 i 是 CUDA 设备序号。 使用 PyTorch 等其他流行框架的用户应该熟悉此约定。

可以使用 device 参数通过每个 Warp API 调用显式地定位特定设备：

```python
a = wp.zeros(n, device="cpu")
wp.launch(kernel, dim=a.size, inputs=[a], device="cpu")

b = wp.zeros(n, device="cuda:0")
wp.launch(kernel, dim=b.size, inputs=[b], device="cuda:0")

c = wp.zeros(n, device="cuda:1")
wp.launch(kernel, dim=c.size, inputs=[c], device="cuda:1")


```


**注意**

Warp CUDA 设备（“cuda:i”）对应于设备 i 的主 CUDA 上下文。 这与 PyTorch 等框架和其他使用 CUDA Runtime API 的软件兼容。 它使互操作性变得容易，因为内存等 GPU 资源可以与 Warp 共享。



**class warp.context.Device(runtime, alias, ordinal=-1, is_primary=False, context=None)**

用于分配 Warp 数组并启动内核的设备。

* **ordinal**

    设备的 Warp 特定整数标签。 -1 表示 CPU 设备。

* **name**
    
    设备的字符串标签。 默认情况下，CPU 设备将根据处理器名称命名，如果无法确定处理器名称，则命名为“CPU”。

* **arch**
    
    表示计算能力版本号的整数，计算方式为 10 * 主版本 + 副版本。 0 表示 CPU 设备。

* **is_uva**

    一个布尔值，指示设备是否支持统一寻址。 对于 CPU 设备为 False。

* **is_cubin_supported**

    一个布尔值，指示 Warp 版本的 NVRTC 是否可以直接为该设备的架构生成 CUDA 二进制文件 (cubin)。 对于 CPU 设备为 False。

* **is_mempool_supported**
    
    一个布尔值，指示设备是否支持使用 cuMemAllocAsync 和 cuMemPool 系列 API 进行流排序内存分配。 对于 CPU 设备为 False。

* **is_primary**
    
    一个布尔值，指示该设备的 CUDA 上下文是否也是设备的主要上下文。

* **uuid**

    表示 CUDA 设备的 UUID 的字符串。 UUID 与 nvidia-smi -L 使用的格式相同。 对于 CPU 设备没有。

* **pci_bus_id**

    CUDA设备的字符串标识符，格式为[domain]:[bus]:[device]，其中domain、bus和device都是十六进制值。 对于 CPU 设备没有。

* 属性 **is_cpu**
    
    一个布尔值，指示该设备是否是 CPU 设备。

* 属性 **is_cuda**
    
    指示设备是否为 CUDA 设备的布尔值。

* 属性 **context**
    
    与设备关联的上下文。

* 属性 **has_context**
    
    一个布尔值，指示设备是否具有与其关联的 CUDA 上下文。

* 属性 **stream**
    
    与 CUDA 设备关联的流。

    RAISES：
    RuntimeError – 该设备不是 CUDA 设备。

* 属性 **has_stream**
    
    一个布尔值，指示设备是否具有与其关联的流。

* 属性 **total_memory**

    可用设备内存总量（以字节为单位）。

    该功能目前仅针对 CUDA 设备实现。 如果在 CPU 设备上调用，将返回 0。

* 属性 **free_memory**
    
    根据操作系统，设备上可用的内存量（以字节为单位）。

    该功能目前仅针对 CUDA 设备实现。 如果在 CPU 设备上调用，将返回 0。

## 默认设备
为了简化代码编写，Warp 有默认设备的概念。 当 Warp API 调用中省略设备参数时，将使用默认设备。

在 Warp 初始化期间，如果 CUDA 可用，则默认设备设置为“cuda:0”。 否则，默认设备是“cpu”。

函数 `wp.set_device()` 可用于更改默认设备：

```python
wp.set_device("cpu")
a = wp.zeros(n)
wp.launch(kernel, dim=a.size, inputs=[a])

wp.set_device("cuda:0")
b = wp.zeros(n)
wp.launch(kernel, dim=b.size, inputs=[b])

wp.set_device("cuda:1")
c = wp.zeros(n)
wp.launch(kernel, dim=c.size, inputs=[c])


```
**注意**

对于 CUDA 设备，wp.set_device() 执行两件事：设置 Warp 默认设备并使设备的 CUDA 上下文成为当前设备。 这有助于最大限度地减少针对单个设备的代码块中的 CUDA 上下文切换数量。

对于 PyTorch 用户来说，此函数类似于 torch.cuda.set_device()。 仍然可以在单独的 API 调用中指定不同的设备，如以下代码片段所示：


```python
# set default device
wp.set_device("cuda:0")

# use default device
a = wp.zeros(n)

# use explicit devices
b = wp.empty(n, device="cpu")
c = wp.empty(n, device="cuda:1")

# use default device
wp.launch(kernel, dim=a.size, inputs=[a])

wp.copy(b, a)
wp.copy(c, a)

```

## 范围设备
管理默认设备的另一种方法是使用 wp.ScopedDevice 对象。 它们可以任意嵌套并在退出时恢复以前的默认设备：

```python
with wp.ScopedDevice("cpu"):
    # alloc and launch on "cpu"
    a = wp.zeros(n)
    wp.launch(kernel, dim=a.size, inputs=[a])

with wp.ScopedDevice("cuda:0"):
    # alloc on "cuda:0"
    b = wp.zeros(n)

    with wp.ScopedDevice("cuda:1"):
        # alloc and launch on "cuda:1"
        c = wp.zeros(n)
        wp.launch(kernel, dim=c.size, inputs=[c])

    # launch on "cuda:0"
    wp.launch(kernel, dim=b.size, inputs=[b])

```



**注意**

对于 CUDA 设备，wp.ScopedDevice 使设备的 CUDA 上下文成为当前上下文，并在退出时恢复以前的 CUDA 上下文。 当将 Warp 脚本作为更大管道的一部分运行时，这很方便，因为它避免了更改所包含代码中的 CUDA 上下文的任何副作用。

### 示例：将 wp.ScopedDevice 与多个 GPU 一起使用
以下示例显示如何在所有可用的 CUDA 设备上分配数组和启动内核。
```python
import warp as wp

wp.init()


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


# get all CUDA devices
devices = wp.get_cuda_devices()
device_count = len(devices)

# number of launches
iters = 1000

# list of arrays, one per device
arrs = []

# loop over all devices
for device in devices:
    # use a ScopedDevice to set the target device
    with wp.ScopedDevice(device):
        # allocate array
        a = wp.zeros(250 * 1024 * 1024, dtype=float)
        arrs.append(a)

        # launch kernels
        for _ in range(iters):
            wp.launch(inc, dim=a.size, inputs=[a])

# synchronize all devices
wp.synchronize()

# print results
for i in range(device_count):
    print(f"{arrs[i].device} -> {arrs[i].numpy()}")

```

## 当前 CUDA 设备
Warp 使用设备别名“cuda”来定位当前的 CUDA 设备。 这允许外部代码管理执行 Warp 脚本的 CUDA 设备。 它类似于 PyTorch“cuda”设备，Torch 用户应该熟悉它并简化互操作。

在此代码片段中，我们使用 PyTorch 管理当前 CUDA 设备并调用该设备上的 Warp 内核：

```python
def example_function():
    # create a Torch tensor on the current CUDA device
    t = torch.arange(10, dtype=torch.float32, device="cuda")

    a = wp.from_torch(t)

    # launch a Warp kernel on the current CUDA device
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda")

# use Torch to set the current CUDA device and run example_function() on that device
torch.cuda.set_device(0)
example_function()

# use Torch to change the current CUDA device and re-run example_function() on that device
torch.cuda.set_device(1)
example_function()

```

**注意**

如果代码在代码的另一部分可能不可预测地更改 CUDA 上下文的环境中运行，则使用设备别名“cuda”可能会出现问题。 建议使用“cuda:i”等显式 CUDA 设备来避免此类问题。

## 设备同步
CUDA 内核启动和内存操作可以异步执行。 这允许在不同设备上重叠计算和内存操作。 Warp 允许将主机与特定设备上未完成的异步操作同步：

```python
wp.synchronize_device("cuda:1")

```

wp.synchronize_device() 函数提供比 wp.synchronize() 更细粒度的同步，因为后者等待所有设备完成其工作。

## 自定义 CUDA 上下文
Warp 旨在与任意 CUDA 上下文配合使用，因此它可以轻松集成到不同的工作流程中。

基于 CUDA Runtime API 构建的应用程序针对每个设备的主要上下文。 运行时 API 在幕后隐藏了 CUDA 上下文管理。 在 Warp 中，设备“cuda:i”代表设备 i 的主要上下文，它与 CUDA 运行时 API 一致。

基于 CUDA 驱动程序 API 构建的应用程序直接使用 CUDA 上下文，并且可以在任何设备上创建自定义 CUDA 上下文。 可以使用有利于应用程序的特定关联或互操作功能来创建自定义 CUDA 上下文。 Warp 也可以与这些 CUDA 上下文一起使用。

特殊设备别名“cuda”可用于定位当前 CUDA 上下文，无论这是主上下文还是自定义上下文。

此外，Warp 允许为自定义 CUDA 上下文注册新的设备别名，以便可以通过名称显式定位它们。 如果 CUcontext 指针可用，则可以使用它来创建新的设备别名，如下所示：


```python

wp.map_cuda_device("foo", ctypes.c_void_p(context_ptr))

```

或者，如果应用程序将自定义 CUDA 上下文设置为当前上下文，则可以省略该指针：
```python
wp.map_cuda_device("foo")


```
无论哪种情况，映射自定义 CUDA 上下文都允许我们使用分配的别名直接定位上下文：
```python
with wp.ScopedDevice("foo"):
    a = wp.zeros(n)
    wp.launch(kernel, dim=a.size, inputs=[a])


```

## CUDA 对等访问
如果系统硬件配置支持，CUDA 允许不同 GPU 之间直接进行内存访问。 通常，GPU 应为相同类型，并且可能需要特殊互连（例如 NVLINK 或 PCIe 拓扑）。

在初始化期间，Warp 报告多 GPU 系统是否支持对等访问：
```bash
Warp 0.15.1 initialized:
   CUDA Toolkit 11.5, Driver 12.2
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
     "cuda:1"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
     "cuda:2"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
     "cuda:3"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
   CUDA peer access:
     Supported fully (all-directional)

```

如果消息报告完全支持 CUDA 对等访问，则意味着每个 CUDA 设备都可以访问系统中的所有其他 CUDA 设备。 如果显示“部分支持”，则后面会显示访问矩阵，显示哪些设备可以相互访问。 如果显示“不支持”，则表示不支持任何设备之间的访问。

在代码中，我们可以检查支持并启用对等访问，如下所示：

```python
if wp.is_peer_access_supported("cuda:0", "cuda:1"):
    wp.set_peer_access_enabled("cuda:0", "cuda:1", True):
```

这将允许在设备 cuda:1 上直接访问设备 cuda:0 的内存。 对等访问是定向的，这意味着启用从 cuda:1 访问 cuda:0 不会自动启用从 cuda:0 访问 cuda:1。

启用对等访问的好处是它允许设备之间进行直接内存传输 (DMA)。 这通常是一种更快的数据复制方法，因为否则传输需要使用 CPU 暂存缓冲区来完成。

缺点是启用对等访问会降低分配和释放的性能。 不依赖对等内存传输的程序应禁用此设置。

可以使用作用域管理器临时启用或禁用对等访问：

```python
with wp.ScopedPeerAccess("cuda:0", "cuda:1", True):
    ...

```

**注意**

对等访问不会加速使用 Warp 0.14.0 中引入的流排序内存池分配器分配的数组之间的内存传输。 要加速内存池传输，应启用内存池访问。

* warp.is_peer_access_supported(target_device, peer_device)

    检查peer_device是否可以直接访问该系统上target_device的内存。

    这适用于使用默认 CUDA 分配器分配的内存。 对于使用 CUDA 池化分配器分配的内存，请使用 is_mempool_access_supported()。

    返回：
    一个布尔值，指示系统是否支持此对等访问。

    参数：
    * target_device (设备 | str | 无) –

    * peer_device (设备 | str | 无) –

* warp.is_peer_access_enabled(target_device, peer_device)
    检查peer_device当前是否可以访问target_device的内存。

    这适用于使用默认 CUDA 分配器分配的内存。 对于使用 CUDA 池化分配器分配的内存，请使用 is_mempool_access_enabled()。

    返回：
    一个布尔值，指示当前是否启用此对等访问。

    参数：
    * target_device (设备 | str | 无) –

    * peer_device (设备 | str | 无) –

* warp.set_peer_access_enabled(target_device, peer_device, enable)
    
    启用或禁用从peer_device到target_device内存的直接访问。

    启用对等访问可以提高对等内存传输的速度，但可能会对内存消耗和分配性能产生负面影响。

    这适用于使用默认 CUDA 分配器分配的内存。 对于使用 CUDA 池化分配器分配的内存，请使用 set_mempool_access_enabled()。

    参数：
    * target_device (设备 | str | 无) –

    * peer_device (设备 | str | 无) –

    * enable（布尔）-























































