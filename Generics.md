# NVIDIA CUDA Python编程框架--Warp开发文档第五章: 通用函数
Warp 支持编写通用内核和函数，它们充当可以使用不同具体类型实例化的模板。 这使您可以编写一次代码并以多种数据类型重用它。

## 通用内核
通用内核定义语法与常规内核相同，但您可以使用 types.Any 代替具体类型：

```python
from typing import Any

# generic kernel definition using Any as a placeholder for concrete types
@wp.kernel
def scale(x: wp.array(dtype=Any), s: Any):
    i = wp.tid()
    x[i] = s * x[i]

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = len(data)

x16 = wp.array(data, dtype=wp.float16)
x32 = wp.array(data, dtype=wp.float32)
x64 = wp.array(data, dtype=wp.float64)

# run the generic kernel with different data types
wp.launch(scale, dim=n, inputs=[x16, wp.float16(3)])
wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])
wp.launch(scale, dim=n, inputs=[x64, wp.float64(3)])

print(x16)
print(x32)
print(x64)

```

在幕后，Warp 将自动生成通用内核的新实例以匹配给定的参数类型。

## 类型推断
当启动通用内核时，Warp 会根据参数推断具体类型。 wp.launch() 无需任何特殊语法即可处理通用内核，但我们应该注意作为参数传递的数据类型，以确保推断出正确的类型。

* 标量可以作为常规 Python 数值（例如 42 或 0.5）传递。 Python 整数被解释为 wp.int32，Python 浮点值被解释为 wp.float32。 要指定不同的数据类型并避免歧义，应改用 Warp 数据类型（例如 wp.int64(42) 或 wp.float16(0.5)）。

* 向量和矩阵应作为 Warp 类型而不是元组或列表传递（例如 wp.vec3f(1.0, 2.0, 3.0) 或 wp.mat22h([[1.0, 0.0], [0.0, 1.0]])）。

* Warp 数组和结构体可以正常传递。


## 隐式实例化
当您启动具有一组新数据类型的通用内核时，Warp 会自动创建具有给定类型的该内核的新实例。 这很方便，但这种隐式实例化也有一些缺点。

考虑这三个通用内核启动：

```python
wp.launch(scale, dim=n, inputs=[x16, wp.float16(3)])
wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])
wp.launch(scale, dim=n, inputs=[x64, wp.float64(3)])

```
在每次启动期间，都会生成一个新的内核实例，这会强制重新加载模块。 您可能会在输出中看到类似这样的内容：

```bash
Module __main__ load on device 'cuda:0' took 170.37 ms
Module __main__ load on device 'cuda:0' took 171.43 ms
Module __main__ load on device 'cuda:0' took 179.49 ms
```
这会导致一些潜在的问题：

* 重复重建模块的开销会影响程序的整体性能。

* 较旧的 CUDA 驱动程序不允许在图形捕获期间重新加载模块，这将导致捕获失败。

显式实例化可用于克服这些问题。

## 显式实例化
Warp 允许显式声明具有不同类型的通用内核的实例。 一种方法是使用 @wp.overload 装饰器：

```python
@wp.overload
def scale(x: wp.array(dtype=wp.float16), s: wp.float16):
    ...

@wp.overload
def scale(x: wp.array(dtype=wp.float32), s: wp.float32):
    ...

@wp.overload
def scale(x: wp.array(dtype=wp.float64), s: wp.float64):
    ...

wp.launch(scale, dim=n, inputs=[x16, wp.float16(3)])
wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])
wp.launch(scale, dim=n, inputs=[x64, wp.float64(3)])

```
@wp.overload 装饰器允许重新声明通用内核，而无需重复内核代码。 内核主体只是用省略号 (...) 替换。 Warp 跟踪每个内核的已知重载，因此如果存在重载，它将不会再次实例化。 如果在内核启动之前声明所有重载，则模块将仅在所有内核实例就位时加载一次。

我们还可以使用 wp.overload() 作为函数，以获得稍微更简洁的语法。 我们只需要指定通用内核和具体参数类型的列表：
```python
wp.overload(scale, [wp.array(dtype=wp.float16), wp.float16])
wp.overload(scale, [wp.array(dtype=wp.float32), wp.float32])
wp.overload(scale, [wp.array(dtype=wp.float64), wp.float64])

```
除了参数列表之外，还可以提供字典：
```python
wp.overload(scale, {"x": wp.array(dtype=wp.float16), "s": wp.float16})
wp.overload(scale, {"x": wp.array(dtype=wp.float32), "s": wp.float32})
wp.overload(scale, {"x": wp.array(dtype=wp.float64), "s": wp.float64})

```
出于可读性考虑，字典可能是首选。 使用字典，只需要指定通用参数，当重载某些参数不是通用的内核时，这可以更加简洁。

我们可以轻松地在单个循环中创建重载，如下所示：

```python
for T in [wp.float16, wp.float32, wp.float64]:
    wp.overload(scale, [wp.array(dtype=T), T])

```
最后，wp.overload()函数返回具体的内核实例，可以将其保存在变量中：

```python
scale_f16 = wp.overload(scale, [wp.array(dtype=wp.float16), wp.float16])
scale_f32 = wp.overload(scale, [wp.array(dtype=wp.float32), wp.float32])
scale_f64 = wp.overload(scale, [wp.array(dtype=wp.float64), wp.float64])

```
这些实例被视为常规内核，而不是通用内核。 这意味着启动应该更快，因为 Warp 不需要像启动通用内核时那样从参数推断数据类型。 内核参数的类型要求也比通用内核更宽松，因为 Warp 可以将标量、向量和矩阵转换为已知的所需类型。

```python
# launch concrete kernel instances
wp.launch(scale_f16, dim=n, inputs=[x16, 3])
wp.launch(scale_f32, dim=n, inputs=[x32, 3])
wp.launch(scale_f64, dim=n, inputs=[x64, 3])

```

## 通用函数
与 Warp 内核一样，我们也可以定义通用 Warp 函数：
```python
# generic function
@wp.func
def f(x: Any):
    return x * x

# use generic function in a regular kernel
@wp.kernel
def square_float(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = f(a[i])

# use generic function in a generic kernel
@wp.kernel
def square_any(a: wp.array(dtype=Any)):
    i = wp.tid()
    a[i] = f(a[i])

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = len(data)

af = wp.array(data, dtype=float)
ai = wp.array(data, dtype=int)

# launch regular kernel
wp.launch(square_float, dim=n, inputs=[af])

# launch generic kernel
wp.launch(square_any, dim=n, inputs=[af])
wp.launch(square_any, dim=n, inputs=[ai])

```
通用函数可以在常规和通用内核中使用。 没有必要显式重载泛型函数。 当这些函数在内核中使用时，所有必需的函数重载都会自动生成。

## type() 运算符
考虑以下通用函数：
```python
@wp.func
def triple(x: Any):
    return 3 * x

```
由于 Warp 严格的类型规则，在泛型表达式中使用像 3 这样的数字文字是有问题的。 算术表达式中的操作数必须具有相同的数据类型，但整数文字始终被视为 wp.int32。 如果 x 的数据类型不是 wp.int32，则该函数将无法编译，这意味着它根本不是泛型的。

type() 运算符在这里发挥了作用。 type() 运算符返回其参数的类型，这在预先未知数据类型的泛型函数或内核中非常方便。 我们可以像这样重写该函数，使其适用于更广泛的类型：
```python
@wp.func
def triple(x: Any):
    return type(x)(3) * x

```
type() 运算符对于 Warp 内核和函数中的类型转换非常有用。 例如，这是一个简单的通用 arange() 内核：
```python
@wp.kernel
def arange(a: wp.array(dtype=Any)):
    i = wp.tid()
    a[i] = type(a[0])(i)

n = 10
ai = wp.empty(n, dtype=wp.int32)
af = wp.empty(n, dtype=wp.float32)

wp.launch(arange, dim=n, inputs=[ai])
wp.launch(arange, dim=n, inputs=[af])

```
wp.tid() 返回一个整数，但该值在存储到数组之前会转换为数组的数据类型。 或者，我们可以像这样编写 arange() 内核：

```python
@wp.kernel
def arange(a: wp.array(dtype=Any)):
    i = wp.tid()
    a[i] = a.dtype(i)

```

此变体使用 array.dtype() 运算符，该运算符返回数组内容的类型。

## 局限性
Warp 泛型仍在开发中，并且存在一些限制。

## 模块重新加载行为
正如隐式实例化部分中提到的，启动新的内核重载会触发内核模块的重新编译。 这会增加开销，并且不能很好地适应 Warp 当前的内核缓存策略。 内核缓存依赖于对模块内容进行哈希处理，其中包括迄今为止在 Python 程序中遇到的所有具体内核和函数。 每当添加新内核或通用内核的新实例时，都需要重新加载模块。 重新运行 Python 程序会导致将相同的内核序列添加到模块中，这意味着通用内核的隐式实例化将在每次运行时触发相同的模块重新加载。 这显然并不理想，我们打算在未来改进这种行为。

使用显式实例化通常是一个很好的解决方法，只要在任何内核启动之前以相同的顺序添加重载即可。

请注意，此问题并非特定于通用内核。 如果内核定义与内核启动混合在一起，则向模块添加新的常规内核也可能会触发重复的模块重新加载。 例如：
```python
@wp.kernel
def foo(x: float):
    wp.print(x)

wp.launch(foo, dim=1, inputs=[17])

@wp.kernel
def bar(x: float):
    wp.print(x)

wp.launch(bar, dim=1, inputs=[42])

```
此代码还将在每次内核启动期间触发模块重新加载，即使它根本不使用泛型：
```bash
Module __main__ load on device 'cuda:0' took 155.73 ms
17
Module __main__ load on device 'cuda:0' took 164.83 ms
42

```
## 图形捕捉
在 CUDA 12.2 或更早版本中的图形捕获期间不允许模块重新加载。 内核实例化可以触发模块重新加载，这将导致不支持较新版本 CUDA 的驱动程序上的图形捕获失败。 同样，解决方法是在捕获开始之前显式声明所需的重载。

## 类型变量
Warp 的 type() 运算符在原理上与 Python 的 type() 函数类似，但目前无法在 Warp 内核和函数中使用类型作为变量。 例如，当前不允许以下行为：

```python
@wp.func
def triple(x: Any):
    # TODO:
    T = type(x)
    return T(3) * x

```



























































































