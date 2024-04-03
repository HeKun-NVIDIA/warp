#! https://zhuanlan.zhihu.com/p/690511658

# NVIDIA CUDA Python编程框架--Warp开发文档第四章: 并发

## 异步操作
### 内核加载
在 CUDA 设备上启动的内核相对于主机（CPU Python 线程）是异步的。 启动内核会安排其在 CUDA 设备上的执行，但 `wp.launch()` 函数可以在内核执行完成之前返回。 这允许我们在 CUDA 内核执行时运行一些 CPU 计算，这是在程序中引入并行性的简单方法。

```python
wp.launch(kernel1, dim=n, inputs=[a], device="cuda:0")

# do some CPU work while the CUDA kernel is running
do_cpu_work()

```
在不同 CUDA 设备上启动的内核可以同时执行。 这可用于在不同 GPU 上并行处理独立的子任务，同时使用 CPU 执行其他有用的工作：
```python
# launch concurrent kernels on different devices
wp.launch(kernel1, dim=n, inputs=[a0], device="cuda:0")
wp.launch(kernel2, dim=n, inputs=[a1], device="cuda:1")

# do CPU work while kernels are running on both GPUs
do_cpu_work()

```
目前，在 CPU 上启动内核是同步操作。 换句话说，wp.launch()只有在内核在CPU上执行完毕后才会返回。 要同时运行 CUDA 内核和 CPU 内核，应首先启动 CUDA 内核：

```python
# schedule a kernel on a CUDA device
wp.launch(kernel1, ..., device="cuda:0")

# run a kernel on the CPU while the CUDA kernel is running
wp.launch(kernel2, ..., device="cpu")

```

### 图加载
CUDA 图启动的并发规则与 CUDA 内核启动类似，只是图在 CPU 上不可用。

```python
# capture work on cuda:0 in a graph
with wp.ScopedCapture(device="cuda:0") as capture0:
    do_gpu0_work()

# capture work on cuda:1 in a graph
with wp.ScopedCapture(device="cuda:1") as capture1:
    do_gpu1_work()

# launch captured graphs on the respective devices concurrently
wp.capture_launch(capture0.graph)
wp.capture_launch(capture1.graph)

# do some CPU work while the CUDA graphs are running
do_cpu_work()

```
### 数组创建
创建 CUDA 数组对于主机来说也是异步的。 它涉及在设备上分配内存并对其进行初始化，这是使用内核启动或异步 CUDA memset 操作在后台完成的。

```python
a0 = wp.zeros(n, dtype=float, device="cuda:0")
b0 = wp.ones(n, dtype=float, device="cuda:0")

a1 = wp.empty(n, dtype=float, device="cuda:1")
b1 = wp.full(n, 42.0, dtype=float, device="cuda:1")

```

在此代码片段中，数组 a0 和 b0 在设备 cuda:0 上创建，数组 a1 和 b1 在设备 cuda:1 上创建。 同一设备上的操作是顺序的，但每个设备独立于其他设备执行它们，因此它们可以并发运行。

### 数组复制
在设备之间复制数组也可以是异步的，但有一些细节需要注意。

仅当主机阵列固定时，从主机内存复制到 CUDA 设备以及从 CUDA 设备复制到主机内存才是异步的。 固定内存允许 CUDA 驱动程序使用直接内存传输 (DMA)，这种传输通常速度更快，并且无需 CPU 参与即可完成。 使用固定内存有几个缺点：分配和释放通常较慢，并且系统上可以分配的固定内存数量存在特定于系统的限制。 因此，默认情况下不会固定 Warp CPU 阵列。 您可以在创建 CPU 阵列时通过传递 pinned=True 标志来请求固定分配。 对于用于在主机和设备之间复制数据的阵列来说，这是一个不错的选择，特别是在需要异步传输的情况下。

```python
h = wp.zeros(n, dtype=float, device="cpu")
p = wp.zeros(n, dtype=float, device="cpu", pinned=True)
d = wp.zeros(n, dtype=float, device="cuda:0")

# host-to-device copy
wp.copy(d, h)  # synchronous
wp.copy(d, p)  # asynchronous

# device-to-host copy
wp.copy(h, d)  # synchronous
wp.copy(p, d)  # asynchronous

# wait for asynchronous operations to complete
wp.synchronize_device("cuda:0")

```
同一设备上的 CUDA 阵列之间的复制始终相对于主机异步，因为它不涉及 CPU：
```python
a = wp.zeros(n, dtype=float, device="cuda:0")
b = wp.empty(n, dtype=float, device="cuda:0")

# asynchronous device-to-device copy
wp.copy(a, b)

# wait for transfer to complete
wp.synchronize_device("cuda:0")

```
不同设备上的 CUDA 阵列之间的复制相对于主机也是异步的。 点对点传输需要格外小心，因为 CUDA 设备彼此之间也是异步的。 将数组从一个 GPU 复制到另一个 GPU 时，目标 GPU 用于执行复制，因此我们需要确保源 GPU 上的先前工作在传输之前完成。

```python
a0 = wp.zeros(n, dtype=float, device="cuda:0")
a1 = wp.empty(n, dtype=float, device="cuda:1")

# wait for outstanding work on the source device to complete to ensure the source array is ready
wp.synchronize_device("cuda:0")

# asynchronous peer-to-peer copy
wp.copy(a1, a0)

# wait for the copy to complete on the destination device
wp.synchronize_device("cuda:1")

```

请注意，可以使用内存池访问或对等访问来加速对等传输，这可以在支持的系统上的 CUDA 设备之间实现 DMA 传输。

## 流
CUDA 流是在 GPU 上按顺序执行的操作序列。 来自不同流的操作可以同时运行，并且可以由设备调度器交错运行。

Warp 在初始化期间自动为每个 CUDA 设备创建一个流。 这将成为设备的当前流。 该设备上发出的所有内核启动和内存操作都放在当前流上。

### 创建流
流与特定的 CUDA 设备相关联。 可以使用 wp.Stream 构造函数创建新流：

```python
s1 = wp.Stream("cuda:0")  # create a stream on a specific CUDA device
s2 = wp.Stream()          # create a stream on the default device

```
如果省略 device 参数，将使用默认设备，可以使用 wp.ScopedDevice 进行管理。

为了与外部代码进行互操作，可以传递 CUDA 流句柄来包装外部流：

```python
s3 = wp.Stream("cuda:0", cuda_stream=stream_handle)

```
cuda_stream 参数必须是作为 Python 整数传递的本机流句柄（cudaStream_t 或 CUstream）。 该机制在内部用于与 PyTorch 或 DLPack 等外部框架共享流。 调用者负责确保外部流在被 wp.Stream 对象引用时不会被破坏。

### 使用流
使用 wp.ScopedStream 临时更改设备上的当前流并安排该流上的一系列操作：

```python
stream = wp.Stream("cuda:0")

with wp.ScopedStream(stream):
    a = wp.zeros(n, dtype=float)
    b = wp.empty(n, dtype=float)
    wp.launch(kernel, dim=n, inputs=[a])
    wp.copy(b, a)

```

由于流与特定设备相关联，因此 wp.ScopedStream 包含了 wp.ScopedDevice 的功能。 这就是为什么我们不需要为每个调用显式指定设备参数。

流的一个重要好处是它们可以用于在同一设备上重叠计算和数据传输操作，这可以通过并行执行这些操作来提高程序的整体吞吐量。
```python
with wp.ScopedDevice("cuda:0"):
    a = wp.zeros(n, dtype=float)
    b = wp.empty(n, dtype=float)
    c = wp.ones(n, dtype=float, device="cpu", pinned=True)

    compute_stream = wp.Stream()
    transfer_stream = wp.Stream()

    # asynchronous kernel launch on a stream
    with wp.ScopedStream(compute_stream)
        wp.launch(kernel, dim=a.size, inputs=[a])

    # asynchronous host-to-device copy on another stream
    with wp.ScopedStream(transfer_stream)
        wp.copy(b, c)

```
wp.get_stream() 函数可用于获取设备上的当前流：

```python

s1 = wp.get_stream("cuda:0")  # get the current stream on a specific device
s2 = wp.get_stream()          # get the current stream on the default device
```
wp.set_stream() 函数可用于设置设备上的当前流：

```python
wp.set_stream(stream, device="cuda:0")  # set the stream on a specific device
wp.set_stream(stream)                   # set the stream on the default device

```

一般来说，我们建议使用 wp.ScopedStream 而不是 wp.set_stream()。

### 同步
wp.synchronize_stream() 函数可用于阻塞主机线程，直到给定的流完成：

```python
wp.synchronize_stream(stream)

```
在使用多个流的程序中，这比 wp.synchronize_device() 提供了对同步行为更细粒度的控制，wp.synchronize_device() 会同步设备上的所有流。 例如，如果程序具有多个计算和传输流，则主机可能只想等待一个传输流完成，而不等待其他流。 通过仅同步一个流，我们允许其他流继续与主机线程同时运行。

### 事件
wp.synchronize_device() 或 wp.synchronize_stream() 等函数会阻塞 CPU 线程，直到 CUDA 设备上的工作完成为止，但它们并不旨在使多个 CUDA 流相互同步。

CUDA 事件提供了一种用于流之间的设备端同步的机制。 这种同步不会阻塞主机线程，但它允许一个流等待另一个流上的工作完成。

与流一样，事件与特定设备相关联：

```python
e1 = wp.Event("cuda:0")  # create an event on a specific CUDA device
e2 = wp.Event()          # create an event on the default device

```
为了等待流完成某些工作，我们首先在该流上记录事件。 然后我们让另一个流等待该事件：

```python
stream1 = wp.Stream("cuda:0")
stream2 = wp.Stream("cuda:0")
event = wp.Event("cuda:0")

stream1.record_event(event)
stream2.wait_event(event)

```
请注意，录制事件时，事件必须与录制流来自同一设备。 等待事件时，等待流可以来自另一个设备。 这允许使用事件来同步不同 GPU 上的流。

如果在没有事件参数的情况下调用 record_event() 方法，则会创建、记录并返回一个临时事件：

```python
event = stream1.record_event()
stream2.wait_event(event)

```
wait_stream() 方法将记录和等待事件的行为结合在一次调用中：

```python
stream2.wait_stream(stream1)

```
Warp 还提供了全局函数 wp.record_event()、wp.wait_event() 和 wp.wait_stream()，它们对默认设备的当前流进行操作：
```python
wp.record_event(event)  # record an event on the current stream
wp.wait_event(event)    # make the current stream wait for an event
wp.wait_stream(stream)  # make the current stream wait for another stream

```
这些变体可以方便地在 wp.ScopedStream 和 wp.ScopedDevice 管理器内部使用。

下面是一个更完整的示例，其中包含将数据复制到数组中的生产者流和在内核中使用该数组的消费者流：

```python
with wp.ScopedDevice("cuda:0"):
    a = wp.empty(n, dtype=float)
    b = wp.ones(n, dtype=float, device="cpu", pinned=True)

    producer_stream = wp.Stream()
    consumer_stream = wp.Stream()

    with wp.ScopedStream(producer_stream)
        # asynchronous host-to-device copy
        wp.copy(a, b)

        # record an event to create a synchronization point for the consumer stream
        event = wp.record_event()

        # do some unrelated work in the producer stream
        do_other_producer_work()

    with wp.ScopedStream(consumer_stream)
        # do some unrelated work in the consumer stream
        do_other_consumer_work()

        # wait for the producer copy to complete
        wp.wait_event(event)

        # consume the array in a kernel
        wp.launch(kernel, dim=a.size, inputs=[a])

```
函数 wp.synchronize_event() 可用于阻塞主机线程，直到记录的事件完成。 当主机想要等待流上的特定同步点，同时允许后续流操作继续异步执行时，这非常有用。

```python
with wp.ScopedDevice("cpu"):
    # CPU buffers for readback
    a_host = wp.empty(N, dtype=float, pinned=True)
    b_host = wp.empty(N, dtype=float, pinned=True)

with wp.ScopedDevice("cuda:0"):
    stream = wp.get_stream()

    # initialize first GPU array
    a = wp.full(N, 17, dtype=float)
    # asynchronous readback
    wp.copy(a_host, a)
    # record event
    a_event = stream.record_event()

    # initialize second GPU array
    b = wp.full(N, 42, dtype=float)
    # asynchronous readback
    wp.copy(b_host, b)
    # record event
    b_event = stream.record_event()

    # wait for first array readback to complete
    wp.synchronize_event(a_event)
    # process first array on the CPU
    assert np.array_equal(a_host.numpy(), np.full(N, fill_value=17.0))

    # wait for second array readback to complete
    wp.synchronize_event(b_event)
    # process second array on the CPU
    assert np.array_equal(b_host.numpy(), np.full(N, fill_value=42.0))

```

### CUDA 默认流
Warp 避免使用同步 CUDA 默认流，这是一种与同一设备上的所有其他流同步的特殊流。 该流当前仅在为方便起见而提供的读回操作期间使用，例如 array.numpy() 和 array.list()。

```python
stream1 = wp.Stream("cuda:0")
stream2 = wp.Stream("cuda:0")

with wp.ScopedStream(stream1):
    a = wp.zeros(n, dtype=float)

with wp.ScopedStream(stream2):
    b = wp.ones(n, dtype=float)

print(a)
print(b)

```
在上面的代码片段中，有两个数组在不同的 CUDA 流上初始化。 打印这些数组会触发回读，这是使用 array.numpy() 方法完成的。 此读回发生在同步 CUDA 默认流上，这意味着不需要显式同步。 这样做的原因是方便 - 打印数组对于调试目的很有用，所以最好不用担心同步。

这种方法的缺点是在图形捕获期间无法使用 CUDA 默认流（以及使用它的任何方法）。 常规的 wp.copy() 函数应该用于捕获图中的读回操作。

### 显式流参数
多个 Warp 函数接受可选的流参数。 这允许直接指定流而不使用 wp.ScopedStream 管理器。 这两种方法各有优点和缺点，将在下面讨论。 直接接受流参数的函数包括 wp.launch()、wp.capture_launch() 和 wp.copy()。

要在特定流上启动内核：

```python

wp.launch(kernel, dim=n, inputs=[...], stream=my_stream)
```

当使用显式流参数启动内核时，应省略设备参数，因为设备是从流中推断出来的。 如果同时指定了流和设备，则流参数优先。

要在特定流上启动图表：
```python
wp.capture_launch(graph, stream=my_stream)

```
对于内核和图形启动，直接指定流比使用 wp.ScopedStream 更快。 虽然 wp.ScopedStream 对于在特定流上调度一系列操作非常有用，但在设备上设置和恢复当前流会产生一些开销。 对于较大的工作负载，这种开销可以忽略不计，但性能敏感的代码可能会受益于直接指定流而不是使用 wp.ScopedStream，特别是对于单个内核或图形启动。

除了这些性能考虑因素之外，在两个 CUDA 设备之间复制数组时，直接指定流也很有用。 默认情况下，Warp 使用以下规则来确定哪个流将用于复制：

* 如果目标阵列位于 CUDA 设备上，则使用目标设备上的当前流。

* 否则，如果源数组位于 CUDA 设备上，则使用源设备上的当前流。

在点对点复制的情况下，指定流参数允许覆盖这些规则，并且可以在来自任何设备的流上执行复制。

```python
stream0 = wp.get_stream("cuda:0")
stream1 = wp.get_stream("cuda:1")

a0 = wp.zeros(n, dtype=float, device="cuda:0")
a1 = wp.empty(n, dtype=float, device="cuda:1")

# wait for the destination array to be ready
stream0.wait_stream(stream1)

# use the source device stream to do the copy
wp.copy(a1, a0, stream=stream0)

```
请注意，我们使用事件同步来使源流在复制之前等待目标流。 这是由于 Warp 0.14.0 中引入的流排序内存池分配器。 空数组 a1 的分配被安排在流 stream1 上。 为了避免分配前使用错误，我们需要等到分配完成后再在不同的流上使用该数组。

## 流使用指南
即使对于经验丰富的 CUDA 开发人员来说，流同步也可能是一件棘手的事情。 考虑以下代码：

```python
a = wp.zeros(n, dtype=float, device="cuda:0")

s = wp.Stream("cuda:0")

wp.launch(kernel, dim=a.size, inputs=[a], stream=s)

```
该代码片段存在一个乍一看很难检测到的流同步问题。 代码很可能会正常工作，但它引入了未定义的行为，这可能会导致偶尔出现一次的错误结果。 问题是内核是在流 s 上启动的，这与用于创建数组 a 的流不同。 该数组是在设备 cuda:0 的当前流上分配和初始化的，这意味着当流 s 开始执行使用该数组的内核时，该数组可能尚未准备好。

解决方案是同步流，可以这样完成：

```python
a = wp.zeros(n, dtype=float, device="cuda:0")

s = wp.Stream("cuda:0")

# wait for the current stream on cuda:0 to finish initializing the array
s.wait_stream(wp.get_stream("cuda:0"))

wp.launch(kernel, dim=a.size, inputs=[a], stream=s)

```
wp.ScopedStream 管理器旨在缓解这个常见问题。 它将新流与设备上的先前流同步。 其行为相当于插入 wait_stream() 调用，如上所示。 使用 wp.ScopedStream，我们不需要显式地将新流与前一个流同步：

```python
a = wp.zeros(n, dtype=float, device="cuda:0")

s = wp.Stream("cuda:0")

with wp.ScopedStream(s):
    wp.launch(kernel, dim=a.size, inputs=[a])

```

这使得 wp.ScopedStream 成为在 Warp 中开始使用流的推荐方法。 使用显式流参数可能会稍微提高性能，但需要更多地关注流同步机制。 如果您是流新手，请考虑以下将流集成到 Warp 程序中的轨迹：

* 1 级：不要。 您不需要使用流来使用 Warp。 避免流媒体是一种完全有效且受人尊敬的生活方式。 无需花哨的流处理就可以开发许多有趣且复杂的算法。 通常，最好专注于以简单而优雅的方式解决问题，而不受低级流管理的变幻莫测的阻碍。

* 级别 2：使用 wp.ScopedStream。 它有助于避免一些常见的难以发现的问题。 虽然有一点开销，但如果 GPU 工作负载足够大，开销应该可以忽略不计。 考虑将流添加到程序中作为一种有针对性的优化形式，特别是如果内存传输（“喂养野兽”）等某些领域是已知的瓶颈。 流非常适合重叠内存传输和计算工作负载。

* 第 3 级：使用显式流参数进行内核启动、数组复制等。这将是性能最高的方法，可以让您接近光速。 您需要自己处理所有流同步，但结果在基准测试中可能会很有价值。


## 同步规则
同步的一般规则是尽可能少地使用它，但不能更少。

过度的同步会严重限制程序的性能。 同步意味着流或线程正在等待其他事情完成。 在等待期间，它不会执行任何有用的工作，这意味着在到达同步点之前，任何未完成的工作都无法开始。 这限制了并行执行，而并行执行对于从硬件组件集合中榨取最大收益通常很重要。

另一方面，如果操作无序执行，不充分的同步可能会导致错误或不正确的结果。 如果不能保证正确的结果，那么快速的程序也是没有用的。

### 主机端同步
主机端同步会阻塞主机线程 (Python)，直到 GPU 工作完成。 当您等待某些 GPU 工作完成以便可以访问 CPU 上的结果时，这是必要的。

`wp.synchronize()` 是最繁琐的同步函数，因为它同步系统中的所有设备。 如果性能很重要，那么调用它几乎永远不是正确的函数。 但是，在调试与同步相关的问题时，它有时会很有用。

`wp.synchronize_device(device)` 同步单个设备，通常更好更快。 这将同步指定设备上的所有流，包括 Warp 创建的流和任何其他框架创建的流。

`wp.synchronize_stream(stream)` 同步单个流，这更好。 如果程序使用多个流，您可以等待特定一个流完成，而无需等待其他流。 如果您有一个将数据从 GPU 复制到 CPU 的读回流，这会很方便。 您可以等待传输完成并开始在 CPU 上处理它，而其他流仍在 GPU 上与主机代码并行运行。

`wp.synchronize_event(event)` 是最具体的主机同步函数。 它会阻塞主机，直到先前记录在 CUDA 流上的事件完成。 这可用于等待到达特定的流同步点，同时允许该流上的后续操作异步继续。


### 设备端同步
设备端同步使用 CUDA 事件使一个流等待另一流上记录的同步点（`wp.record_event()、wp.wait_event()、wp.wait_stream()`）。

这些函数不会阻塞主机线程，因此 CPU 可以保持忙于做有用的工作，例如准备下一批数据来喂养野兽。 事件可用于同步同一设备甚至不同 CUDA 设备上的流，因此您可以编排完全在可用 GPU 上执行的非常复杂的多流和多设备工作负载。 这允许将主机端同步保持在最低限度，也许仅在读回最终结果时。

### 同步和图形捕获
CUDA 图捕获 CUDA 流上的一系列操作，可以以较低的开销多次重放。 在捕获期间，不允许使用某些 CUDA 功能，其中包括主机端同步功能。 也不允许使用同步 CUDA 默认流。 CUDA 图中允许的唯一同步形式是基于事件的同步。

CUDA 图形捕获必须在同一流上开始和结束，但中间可以使用多个流。 这使得 CUDA 图形能够包含多个流，甚至多个 GPU。 事件在多流图捕获中发挥着至关重要的作用，因为除了常规同步职责之外，它们还用于将新流分叉和加入到主捕获流中。

以下是在每个设备上使用流捕获多 GPU 图形的示例：

```python
stream0 = wp.Stream("cuda:0")
stream1 = wp.Stream("cuda:1")

# use stream0 as the main capture stream
with wp.ScopedCapture(stream=stream0) as capture:

    # fork stream1, which adds it to the set of streams being captured
    stream1.wait_stream(stream0)

    # launch a kernel on stream0
    wp.launch(kernel, ..., stream=stream0)

    # launch a kernel on stream1
    wp.launch(kernel, ..., stream=stream1)

    # join stream1
    stream0.wait_stream(stream1)

# launch the multi-GPU graph, which can execute the captured kernels concurrently
wp.capture_launch(capture.graph)

```





























