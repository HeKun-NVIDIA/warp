# NVIDIA CUDA Python编程框架--Warp开发文档--Allocators

## 流顺序内存池分配器
### 介绍
Warp 0.14.0 添加了对 CUDA 数组的流排序内存池分配器的支持。 从 Warp 0.15.0 开始，这些分配器在所有支持它们的 CUDA 设备上默认启用。 “流顺序内存池分配器”很拗口，所以让我们一次一点地解开它。

每当创建数组时，都需要在设备上分配内存：
```python
a = wp.empty(n, dtype=float, device="cuda:0")
b = wp.zeros(n, dtype=float, device="cuda:0")
c = wp.ones(n, dtype=float, device="cuda:0")
d = wp.full(n, 42.0, dtype=float, device="cuda:0")

```

上面的每个调用都会分配一块足够大的设备内存来容纳数组，并可以选择使用指定的值初始化内容。 wp.empty() 是唯一一个不以任何方式初始化内容的函数，它只是分配内存。

内存池分配器从更大的保留内存池中获取一块内存，这通常比向操作系统请求全新的存储块要快。 这是这些池化分配器的一个重要好处——它们更快。

流排序意味着每个分配都安排在 CUDA 流上，它表示在 GPU 上按顺序执行的指令序列。 主要好处是它允许在 CUDA 图中分配内存，这在以前是不可能的：

```python
with wp.ScopedCapture() as capture:
    a = wp.zeros(n, dtype=float)
    wp.launch(kernel, dim=a.size, inputs=[a])

wp.capture_launch(capture.graph)

```
从现在开始，我们将这些分配器简称为内存池分配器。

### 配置
Mempool 分配器是 CUDA 的一项功能，大多数现代设备和操作系统都支持该功能。 但是，可能存在不支持它们的系统，例如某些虚拟机设置。 Warp 的设计考虑到了弹性，因此在引入这些新分配器之前编写的现有代码应该继续运行，无论底层系统是否支持它们。

Warp 的启动消息给出了这些分配器的状态，例如：

```bash
Warp 0.15.1 initialized:
CUDA Toolkit 11.5, Driver 12.2
Devices:
    "cpu"      : "x86_64"
    "cuda:0"   : "NVIDIA GeForce RTX 4090" (24 GiB, sm_89, mempool enabled)
    "cuda:1"   : "NVIDIA GeForce RTX 3090" (24 GiB, sm_86, mempool enabled)
```

请注意每个 CUDA 设备旁边的启用内存池的文本。 这意味着设备上启用了内存池。 每当您在该设备上创建数组时，都会使用内存池分配器对其进行分配。 如果您看到内存池受支持，则意味着内存池受支持，但在启动时未启用。 如果您看到 mempool not support，则意味着内存池无法在此设备上使用。

有一个配置标志控制是否应在 wp.init() 期间自动启用内存池：

```python
import warp as wp

wp.config.enable_mempools_at_init = False

wp.init()

```
该标志默认为 True，但如果需要，可以设置为 False。 调用 `wp.init()` 后更改此配置标志无效。

在 `wp.init()` 之后，您可以检查每个设备上是否启用了内存池，如下所示：

```python
if wp.is_mempool_enabled("cuda:0"):
    ...

```

您还可以独立控制每个设备上的启用：
```python
if wp.is_mempool_supported("cuda:0"):
    wp.set_mempool_enabled("cuda:0", True)

```
可以使用作用域管理器临时启用或禁用内存池：
```python
with wp.ScopedMempool("cuda:0", True):
    a = wp.zeros(n, dtype=float, device="cuda:0")

with wp.ScopedMempool("cuda:0", False):
    b = wp.zeros(n, dtype=float, device="cuda:0")

```
在上面的代码片段中，数组 a 将使用内存池分配器进行分配，数组 b 将使用默认分配器进行分配。

在大多数情况下，没有必要摆弄这些启用功能，但如果您需要它们，它们就在那里。 默认情况下，Warp 将在启动时启用内存池（如果支持），这将带来自动提高分配速度的好处。 大多数 Warp 代码应该在有或没有内存池分配器的情况下继续运行，但图形捕获期间的内存分配除外，如果未启用内存池，这将引发异常。

* warp.is_mempool_supported(device)
  
    检查设备上 CUDA 内存池分配器是否可用。

    PARAMETERS:
    device (Device | str | None) –

* warp.is_mempool_enabled(device)

    检查设备上是否启用了 CUDA 内存池分配器。

    PARAMETERS:
    device (Device | str | None) –

* warp.set_mempool_enabled(device, enable)

    启用或禁用设备上的 CUDA 内存池分配器。

    池化分配器通常速度更快，并且允许在图形捕获期间分配内存。

    通常应该启用它们，但有一个罕见的警告。 如果使用池化分配器分配内存并且两个 GPU 之间未启用内存池访问，则在图形捕获期间在不同 GPU 之间复制数据可能会失败。 这是与 Warp 无关的内部 CUDA 限制。 首选解决方案是使用 warp.set_mempool_access_enabled() 启用内存池访问。 如果不支持对等访问，则必须在图形捕获之前使用默认的 CUDA 分配器来预分配内存。

    PARAMETERS:
    device (Device | str | None) –

    enable (bool) –

### 分配执行
分配和释放内存是相当昂贵的操作，会增加程序的开销。 我们无法避免它们，因为我们需要在某个地方为数据分配存储空间，但是有一些简单的策略可以减少分配对性能的总体影响。

考虑以下示例：
```python
for i in range(100):
    a = wp.zeros(n, dtype=float, device="cuda:0")
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

```

在循环的每次迭代中，我们分配一个数组并对数据运行一个内核。 该程序有 100 次分配和 100 次释放。 当我们为 a 分配一个新值时，之前的值会被 Python 收集为垃圾，从而触发释放。

### 重用内存
如果数组的大小保持固定，请考虑在后续迭代中重用内存。 我们只能分配数组一次，并在每次迭代时重新初始化其内容：
```python
# pre-allocate the array
a = wp.empty(n, dtype=float, device="cuda:0")
for i in range(100):
    # reset the contents
    a.zero_()
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

```
如果数组大小在每次迭代中都没有改变，那么这种方法就很有效。 如果大小发生变化但上限已知，我们仍然可以预先分配一个足够大的缓冲区来存储任何迭代中的所有元素。

```python
# pre-allocate a big enough buffer
buffer = wp.empty(MAX_N, dtype=float, device="cuda:0")
for i in range(100):
    # get a buffer slice of size n <= MAX_N
    n = get_size(i)
    a = buffer[:n]
    # reset the contents
    a.zero_()
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

```
以这种方式重用内存可以提高性能，但也可能会给我们的代码增加不必要的复杂性。 内存池分配器有一个有用的功能，可以提高分配性能，而无需以任何方式修改我们的原始代码。

### 释放阈值
内存池释放阈值确定分配器在将其释放回操作系统之前应保留多少保留内存。 对于频繁分配和释放内存的程序，设置较高的释放阈值可以提高分配的性能。

默认情况下，释放阈值设置为 0。如果先前已获取内存并将其返回到池中，则将其设置为更高的数字将减少分配成本。

```python
# set the release threshold to reduce re-allocation overhead
wp.set_mempool_release_threshold("cuda:0", 1024**3)

for i in range(100):
    a = wp.zeros(n, dtype=float, device="cuda:0")
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

```
0 到 1 之间的阈值被解释为可用内存的分数。 例如，0.5 表示设备物理内存的一半，1.0 表示全部内存。 较大的值被解释为绝对字节数。 例如，1024**3 表示 1 GiB 内存。

这是一个简单的优化，可以在不以任何方式修改现有代码的情况下提高程序的性能。

* warp.set_mempool_release_threshold(device, threshold)

    设置设备上的 CUDA 内存池释放阈值。

    这是在尝试将内存释放回操作系统之前要保留的保留内存量。 当内存池持有的字节数超过此数量时，分配器将在下次调用流、事件或设备同步时尝试将内存释放回操作系统。

    0 到 1 之间的值被解释为可用内存的分数。 例如，0.5 表示设备物理内存的一半。 较大的值被解释为绝对字节数。 例如，1024**3 表示 1 GiB 内存。

    PARAMETERS:

    * device (Device | str | None) –

    * threshold (int | float) –


## 图分配
Mempool 分配器可以在 CUDA 图中使用，这意味着您可以捕获创建数组的 Warp 代码：
```python
with wp.ScopedCapture() as capture:
    a = wp.full(n, 42, dtype=float)

wp.capture_launch(capture.graph)

print(a)

```

捕获分配类似于捕获其他操作，例如内核启动或内存复制。 在捕获期间，操作实际上并不执行，而是被记录下来。 要执行捕获的操作，我们必须使用 wp.capture_launch() 启动图形。 如果您想使用在图形捕获期间分配的数组，请记住这一点很重要。 在捕获的图表启动之前，该数组实际上并不存在。 在上面的代码片段中，如果我们在调用 wp.capture_launch() 之前尝试打印数组，则会收到错误。

更一般地说，在图形捕获期间分配内存的能力大大增加了图形中可以捕获的代码范围。 这包括创建临时分配的任何代码。 CUDA 图可用于以最小的 CPU 开销重新运行操作，从而显着提高性能。

## 内存池访问
在支持对等访问的多 GPU 系统上，我们可以从不同的设备直接访问内存池：

```python
if wp.is_mempool_access_supported("cuda:0", "cuda:1"):
    wp.set_mempool_access_enabled("cuda:0", "cuda:1", True):

```
这将允许在设备 cuda:1 上直接访问设备 cuda:0 的内存池。 内存池访问是定向的，这意味着启用从 cuda:1 访问 cuda:0 不会自动启用从 cuda:0 访问 cuda:1。

启用内存池访问的好处是它允许设备之间进行直接内存传输 (DMA)。 这通常是一种更快的数据复制方法，因为否则传输需要使用 CPU 暂存缓冲区来完成。

缺点是启用内存池访问会稍微降低分配和释放的性能。 然而，对于依赖于在设备之间复制内存的应用程序来说，应该有一个净收益。

可以使用作用域管理器临时启用或禁用内存池访问：
```python
with wp.ScopedMempoolAccess("cuda:0", "cuda:1", True):
    a0 = wp.zeros(n, dtype=float, device="cuda:0")
    a1 = wp.empty(n, dtype=float, device="cuda:1")

    # use direct memory transfer between GPUs
    wp.copy(a1, a0)

```
请注意，内存池访问仅适用于使用内存池分配器分配的内存。 对于使用默认 CUDA 分配器分配的内存，我们可以启用 CUDA 对等访问以获得类似的好处。

由于启用内存池访问可能有缺点，因此即使支持，Warp 也不会自动启用它。 因此，不需要在 GPU 之间复制数据的程序不会受到任何影响。


* warp.is_mempool_access_supported(target_device, peer_device)

    检查peer_device是否可以直接访问target_device的内存池。

    如果可以访问内存池，则可以使用 set_mempool_access_enabled() 和 is_mempool_access_enabled() 进行管理。

    返回：
    一个布尔值，指示系统是否支持此内存池访问。

    PARAMETERS:
    * target_device (Device | str | None) –

    * peer_device (Device | str | None) –


* warp.is_mempool_access_enabled(target_device, peer_device)

    检查peer_device当前是否可以访问target_device的内存池。

    这适用于使用 CUDA 池分配器分配的内存。 对于使用默认 CUDA 分配器分配的内存，请使用 is_peer_access_enabled()。

    返回：一个布尔值，指示当前是否启用此对等访问。

    PARAMETERS:
    * target_device (Device | str | None) –

    * peer_device (Device | str | None) –

* warp.set_mempool_access_enabled(target_device, peer_device, enable)

    启用或禁用peer_device对target_device内存池的访问。

    这适用于使用 CUDA 池分配器分配的内存。 对于使用默认 CUDA 分配器分配的内存，请使用 set_peer_access_enabled()。

    PARAMETERS:
    * target_device (Device | str | None) –

    * peer_device (Device | str | None) –

    * enable (bool) –

## 局限性
### 图形捕获期间 GPU 之间的 Mempool 到 Mempool 复制
如果使用内存池分配器分配源和目标，并且设备之间未启用内存池访问，则在图形捕获期间在不同 GPU 之间复制数据将失败。 请注意，这仅适用于捕获图中内存池到内存池的副本； 在图形捕获之外完成的复制不受影响。 同一内存池（即同一设备）内的副本也不受影响。

有两种解决方法。 如果支持内存池访问，您只需在图形捕获之前启用设备之间的内存池访问，如内存池访问中所示。

如果不支持内存池访问，您将需要使用默认的 CUDA 分配器预先分配复制中涉及的数组。 这需要在捕获开始之前完成：

```python

# pre-allocate the arrays with mempools disabled
with wp.ScopedMempool("cuda:0", False):
    a0 = wp.zeros(n, dtype=float, device="cuda:0")
with wp.ScopedMempool("cuda:1", False):
    a1 = wp.empty(n, dtype=float, device="cuda:1")

with wp.ScopedCapture("cuda:1") as capture:
    wp.copy(a1, a0)

wp.capture_launch(capture.graph)
```















