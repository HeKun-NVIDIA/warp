#! https://zhuanlan.zhihu.com/p/690217330
# NVIDIA CUDA Python编程框架--Warp开发文档
Warp 是一个用于编写高性能模拟和图形代码的 Python 框架。 Warp 采用常规 Python 函数，JIT 将它们编译为可以在 CPU 或 GPU 上运行的高效内核代码。

Warp 专为空间计算而设计，并附带一组丰富的原语，可以轻松编写物理模拟、感知、机器人和几何处理程序。 此外，Warp 内核是可微分的，可以用作 PyTorch 和 JAX 等框架的机器学习管道的一部分。

以下是使用 Warp 实现的模拟的一些示例：
![](https://nvidia.github.io/warp/_images/header.jpg)

## 快速开始
Warp 支持 Python 3.7 及以上版本。 它可以在 Windows、Linux 和 macOS 上的 x86-64 和 ARMv8 CPU 上运行。 GPU 支持需要支持 CUDA 的 NVIDIA GPU 和驱动程序（最低 GeForce GTX 9xx）。

安装 Warp 最简单的方法是通过 PyPI：
```bash
$ pip install warp-lang
```

发布页面上还提供了预构建的二进制包。 要在本地 Python 环境中安装，请提取存档并从根目录运行以下命令：
```bash
$ pip install .
```

## 基本示例
下面给出了计算随机 3D 向量长度的第一个程序示例：

```bash
import warp as wp
import numpy as np

wp.init()

num_points = 1024

@wp.kernel
def length(points: wp.array(dtype=wp.vec3),
           lengths: wp.array(dtype=float)):

    # thread index
    tid = wp.tid()

    # compute distance of each point from origin
    lengths[tid] = wp.length(points[tid])


# allocate an array of 3d points
points = wp.array(np.random.rand(num_points, 3), dtype=wp.vec3)
lengths = wp.zeros(num_points, dtype=float)

# launch kernel
wp.launch(kernel=length,
          dim=len(points),
          inputs=[points, lengths])

print(lengths)
```

## 其他示例
Github 存储库中的示例目录包含许多脚本，这些脚本展示了如何使用 Warp API 实现不同的模拟方法。 大多数示例将在与示例相同的目录中生成包含时间采样动画的 USD 文件。 在运行示例之前，用户应确保使用以下命令安装了 usd-core 软件包：
```bash
pip install usd-core
```

可以从命令行运行示例，如下所示：

```bash
python -m warp.examples.<example_subdir>.<example>
```
大多数示例可以在 CPU 或支持 CUDA 的设备上运行，但少数示例需要支持 CUDA 的设备。 这些标记在示例脚本的顶部。

USD 文件可以在 NVIDIA Omniverse、Pixar 的 UsdView 和 Blender 中查看或渲染。 请注意，不建议在 macOS 中使用预览，因为它对时间采样动画的支持有限。

内置单元测试可以从命令行运行，如下所示：

```bash
python -m warp.tests
```
**注意: 下方示例点击图片可以直接跳转到示例代码**

**在以后章节中会逐步更新编程细节**

<h3>examples/core<a class="headerlink" href="#examples-core" title="Link to this heading">#</a></h3>
<div class="table-wrapper gallery docutils container">
<table class="gallery docutils align-default">
<tbody>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_dem.py"><img alt="_images/core_dem.png" src="https://nvidia.github.io/warp/_images/core_dem.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_fluid.py"><img alt="_images/core_fluid.png" src="https://nvidia.github.io/warp/_images/core_fluid.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_graph_capture.py"><img alt="_images/core_graph_capture.png" src="https://nvidia.github.io/warp/_images/core_graph_capture.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_marching_cubes.py"><img alt="_images/core_marching_cubes.png" src="https://nvidia.github.io/warp/_images/core_marching_cubes.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>dem</p></td>
<td><p>fluid</p></td>
<td><p>graph capture</p></td>
<td><p>marching cubes</p></td>
</tr>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_mesh.py"><img alt="_images/core_mesh.png" src="https://nvidia.github.io/warp/_images/core_mesh.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_nvdb.py"><img alt="_images/core_nvdb.png" src="https://nvidia.github.io/warp/_images/core_nvdb.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raycast.py"><img alt="_images/core_raycast.png" src="https://nvidia.github.io/warp/_images/core_raycast.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raymarch.py"><img alt="_images/core_raymarch.png" src="https://nvidia.github.io/warp/_images/core_raymarch.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>mesh</p></td>
<td><p>nvdb</p></td>
<td><p>raycast</p></td>
<td><p>raymarch</p></td>
</tr>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sph.py"><img alt="_images/core_sph.png" src="https://nvidia.github.io/warp/_images/core_sph.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_torch.py"><img alt="_images/core_torch.png" src="https://nvidia.github.io/warp/_images/core_torch.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_wave.py"><img alt="_images/core_wave.png" src="https://nvidia.github.io/warp/_images/core_wave.png" /></a>
</td>
<td></td>
</tr>
<tr class="row-even"><td><p>sph</p></td>
<td><p>torch</p></td>
<td><p>wave</p></td>
<td></td>
</tr>
</tbody>
</table>
</div>


<h3>examples/fem<a class="headerlink" href="#examples-fem" title="Link to this heading">#</a></h3>
<div class="table-wrapper gallery docutils container">
<table class="gallery docutils align-default">
<tbody>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_apic_fluid.py"><img alt="_images/fem_apic_fluid.png" src="https://nvidia.github.io/warp/_images/fem_apic_fluid.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_convection_diffusion.py"><img alt="_images/fem_convection_diffusion.png" src="https://nvidia.github.io/warp/_images/fem_convection_diffusion.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion_3d.py"><img alt="_images/fem_diffusion_3d.png" src="https://nvidia.github.io/warp/_images/fem_diffusion_3d.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion.py"><img alt="_images/fem_diffusion.png" src="https://nvidia.github.io/warp/_images/fem_diffusion.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>apic fluid</p></td>
<td><p>convection diffusion</p></td>
<td><p>diffusion 3d</p></td>
<td><p>diffusion</p></td>
</tr>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_mixed_elasticity.py"><img alt="_images/fem_mixed_elasticity.png" src="https://nvidia.github.io/warp/_images/fem_mixed_elasticity.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_navier_stokes.py"><img alt="_images/fem_navier_stokes.png" src="https://nvidia.github.io/warp/_images/fem_navier_stokes.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_stokes_transfer.py"><img alt="_images/fem_stokes_transfer.png" src="https://nvidia.github.io/warp/_images/fem_stokes_transfer.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_stokes.py"><img alt="_images/fem_stokes.png" src="https://nvidia.github.io/warp/_images/fem_stokes.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>mixed elasticity</p></td>
<td><p>navier stokes</p></td>
<td><p>stokes transfer</p></td>
<td><p>stokes</p></td>
</tr>
</tbody>
</table>
</div>



<h3>examples/optim<a class="headerlink" href="#examples-optim" title="Link to this heading">#</a></h3>
<div class="table-wrapper gallery docutils container">
<table class="gallery docutils align-default">
<tbody>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_bounce.py"><img alt="_images/optim_bounce.png" src="https://nvidia.github.io/warp/_images/optim_bounce.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_cloth_throw.py"><img alt="_images/optim_cloth_throw.png" src="https://nvidia.github.io/warp/_images/optim_cloth_throw.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_diffray.py"><img alt="_images/optim_diffray.png" src="https://nvidia.github.io/warp/_images/optim_diffray.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_drone.py"><img alt="_images/optim_drone.png" src="https://nvidia.github.io/warp/_images/optim_drone.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>bounce</p></td>
<td><p>cloth throw</p></td>
<td><p>diffray</p></td>
<td><p>drone</p></td>
</tr>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_inverse_kinematics.py"><img alt="_images/optim_inverse_kinematics.png" src="https://nvidia.github.io/warp/_images/optim_inverse_kinematics.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_spring_cage.py"><img alt="_images/optim_spring_cage.png" src="https://nvidia.github.io/warp/_images/optim_spring_cage.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_trajectory.py"><img alt="_images/optim_trajectory.png" src="https://nvidia.github.io/warp/_images/optim_trajectory.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_walker.py"><img alt="_images/optim_walker.png" src="https://nvidia.github.io/warp/_images/optim_walker.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>inverse kinematics</p></td>
<td><p>spring cage</p></td>
<td><p>trajectory</p></td>
<td><p>walker</p></td>
</tr>
</tbody>
</table>
</div>


<h3>examples/sim<a class="headerlink" href="#examples-sim" title="Link to this heading">#</a></h3>
<div class="table-wrapper gallery docutils container">
<table class="gallery docutils align-default">
<tbody>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_cartpole.py"><img alt="_images/sim_cartpole.png" src="https://nvidia.github.io/warp/_images/sim_cartpole.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_cloth.py"><img alt="_images/sim_cloth.png" src="https://nvidia.github.io/warp/_images/sim_cloth.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_granular.py"><img alt="_images/sim_granular.png" src="https://nvidia.github.io/warp/_images/sim_granular.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_granular_collision_sdf.py"><img alt="_images/sim_granular_collision_sdf.png" src="https://nvidia.github.io/warp/_images/sim_granular_collision_sdf.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>cartpole</p></td>
<td><p>cloth</p></td>
<td><p>granular</p></td>
<td><p>granular collision sdf</p></td>
</tr>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_jacobian_ik.py"><img alt="_images/sim_jacobian_ik.png" src="https://nvidia.github.io/warp/_images/sim_jacobian_ik.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_quadruped.py"><img alt="_images/sim_quadruped.png" src="https://nvidia.github.io/warp/_images/sim_quadruped.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_chain.py"><img alt="_images/sim_rigid_chain.png" src="https://nvidia.github.io/warp/_images/sim_rigid_chain.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_contact.py"><img alt="_images/sim_rigid_contact.png" src="https://nvidia.github.io/warp/_images/sim_rigid_contact.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>jacobian ik</p></td>
<td><p>quadruped</p></td>
<td><p>rigid chain</p></td>
<td><p>rigid contact</p></td>
</tr>
<tr class="row-odd"><td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_force.py"><img alt="_images/sim_rigid_force.png" src="https://nvidia.github.io/warp/_images/sim_rigid_force.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_gyroscopic.py"><img alt="_images/sim_rigid_gyroscopic.png" src="https://nvidia.github.io/warp/_images/sim_rigid_gyroscopic.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_soft_contact.py"><img alt="_images/sim_rigid_soft_contact.png" src="https://nvidia.github.io/warp/_images/sim_rigid_soft_contact.png" /></a>
</td>
<td><a class="reference external image-reference" href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_soft_body.py"><img alt="_images/sim_soft_body.png" src="https://nvidia.github.io/warp/_images/sim_soft_body.png" /></a>
</td>
</tr>
<tr class="row-even"><td><p>rigid force</p></td>
<td><p>rigid gyroscopic</p></td>
<td><p>rigid soft contact</p></td>
<td><p>soft body</p></td>
</tr>
</tbody>
</table>
</div>








