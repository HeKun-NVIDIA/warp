#! https://zhuanlan.zhihu.com/p/690217271
# NVIDIA CUDA Python编程框架--Warp开发文档: Basics

## 初始化
在使用 Warp 之前，应使用 wp.init() 方法显式初始化，如下所示：
```python
import warp as wp

wp.init()

```
Warp 将打印一些有关可用计算设备、驱动程序版本以及任何生成的内核代码的位置的启动信息，例如：

```bash
Warp 1.0.0 initialized:
CUDA Toolkit: 11.8, Driver: 12.1
Devices:
    "cpu"    | AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD
    "cuda:0" | NVIDIA GeForce RTX 4080 (sm_89)
Kernel cache: C:\Users\mmacklin\AppData\Local\NVIDIA\warp\Cache\1.0.0
```

## 内核
在 Warp 中，计算内核被定义为 Python 函数，并使用 `@wp.kernel` 装饰器进行注释，如下所示：

```bash
@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # compute the dot product between vectors
    r = wp.dot(x, y)

    # write result back to memory
    c[tid] = r
```
由于 Warp 内核被编译为本机 C++/CUDA 代码，因此所有函数输入参数都必须是静态类型的。 这使得 Warp 能够生成以本质上本机速度执行的快速代码。 由于内核可以在 CPU 或 GPU 上运行，因此它们无法从 Python 环境访问任意全局状态。 相反，它们必须通过输入参数（例如数组）读取和写入数据。

Warp 内核函数与 CUDA 内核具有 1:1 对应关系，要启动具有 1024 个线程的内核，我们使用 wp.launch() 如下：

```python
wp.launch(kernel=simple_kernel, # kernel to launch
          dim=1024,             # number of threads
          inputs=[a, b, c],     # parameters
          device="cuda")        # execution device

```

在内核内部，我们可以使用 wp.tid() 内置函数检索每个线程的线程索引：

```python
# get thread index
i = wp.tid()

```
内核可以使用 1D、2D、3D 或 4D 线程网格启动，例如：要启动 2D 线程网格来处理 1024x1024 图像，我们可以编写：
```python
wp.launch(kernel=compute_image,
          dim=(1024, 1024),
          inputs=[img],
          device="cuda")

```
然后，在内核内部我们可以检索 2D 线程索引，如下所示：

```python
# get thread index
i, j = wp.tid()

# write out a color value for each pixel
color[i, j] = wp.vec3(r, g, b)

```
## 示例：更改内核缓存目录
以下示例说明了如何在调用 wp.init() 之前和之后更改生成和编译的内核代码的位置。

```python
import os

import warp as wp

example_dir = os.path.dirname(os.path.realpath(__file__))

# set default cache directory before wp.init()
wp.config.kernel_cache_dir = os.path.join(example_dir, "tmp", "warpcache1")

wp.init()

print("+++ Current cache directory: ", wp.config.kernel_cache_dir)

# change cache directory after wp.init()
wp.build.init_kernel_cache(os.path.join(example_dir, "tmp", "warpcache2"))

print("+++ Current cache directory: ", wp.config.kernel_cache_dir)

# clear kernel cache (forces fresh kernel builds every time)
wp.build.clear_kernel_cache()


@wp.kernel
def basic(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = float(tid)


device = "cpu"
n = 10
x = wp.zeros(n, dtype=float, device=device)

wp.launch(kernel=basic, dim=n, inputs=[x], device=device)
print(x.numpy())

```


## 数组
内存分配通过 wp.array 类型公开。 数组包装了可能位于主机 (CPU) 或设备 (GPU) 内存中的底层内存分配。 数组是强类型的，存储内置值的线性序列（float、int、vec3、matrix33 等）。

数组的分配方式与 PyTorch 类似：

```python
# allocate an uninitialized array of vec3s
v = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

# allocate a zero-initialized array of quaternions
q = wp.zeros(shape=n, dtype=wp.quat, device="cuda")

# allocate and initialize an array from a NumPy array
# will be automatically transferred to the specified device
a = np.ones((10, 3), dtype=np.float32)
v = wp.from_numpy(a, dtype=wp.vec3, device="cuda")

```
默认情况下，从外部数据（例如：NumPy、列表、元组）初始化的 Warp 数组将为指定设备创建数据副本到新内存。 但是，只要输入是连续的并且位于同一设备上，数组就可以使用数组构造函数的 `copy=False `参数来别名外部存储器。 有关与外部框架共享内存的更多详细信息，请参阅[互操作性](https://nvidia.github.io/warp/modules/interoperability.html)部分。

要将 GPU 数组数据读回 CPU 内存，我们可以使用 array.numpy() 方法：
```python
# bring data from device back to host
view = device_array.numpy()
```

这将自动与 GPU 同步，以确保所有未完成的工作已完成，并将数组复制回 CPU 内存，并在其中传递给 NumPy。 在 CPU 阵列上调用 array.numpy() 将返回 Warp 数据的零拷贝 NumPy 视图。

## 用户函数
用户可以使用@wp.func装饰器编写自己的函数，例如：
```python
@wp.func
def square(x: float):
    return x*x

```
用户函数可以在同一模块内的内核中自由调用，并接受数组作为输入。

## 编译模型
Warp 使用 `Python->C++/CUDA` 编译模型，从 Python 函数定义生成内核代码。 属于 Python 模块的所有内核都会在运行时编译为动态库和 PTX，然后在应用程序重新启动之间缓存结果，以实现快速启动。

请注意，编译是在该模块第一次内核启动时触发的。 使用 @wp.kernel 在模块中注册的任何内核都将包含在共享库中。
![](https://nvidia.github.io/warp/_images/compiler_pipeline.png)


## 语言详情
为了支持 GPU 计算和可微分性，与 CPython 运行时存在一些差异。

## 内置类型
Warp 支持许多类似于高级着色语言的内置数学类型，例如 `vec2、vec3、vec4、mat22、mat33、mat44、quat、array`。 所有内置类型都具有值语义，因此 a = b 等表达式会生成变量 b 的副本而不是引用。

## 强类型
与 Python 不同，在 Warp 中所有变量都必须输入类型。 类型是使用 Python 类型扩展从源表达式和函数签名推断出来的。 所有内核参数都必须使用适当的类型进行注释，例如：

```python
@wp.kernel
def simple_kernel(a: wp.array(dtype=vec3),
                  b: wp.array(dtype=vec3),
                  c: float):

```
不支持元组初始化，而是应显式键入变量：

```python
# invalid
a = (1.0, 2.0, 3.0)

# valid
a = wp.vec3(1.0, 2.0, 3.0)
```































