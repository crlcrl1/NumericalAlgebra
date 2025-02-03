# 数值代数大作业

> 陈润璘 2200010848

## 在 CPU 上运行 C++ 代码

### 运行环境

大作业中的 C++ 代码使用了 [Eigen](https://eigen.tuxfamily.org) 线性代数库，版本为 3.4.0，因此需要先安装 Eigen。为了在多核处理器上运行，我们使用了 OpenMP，因此需要确保编译器支持 OpenMP。

### 编译代码

安装 Eigen 后，可以在 `cpp/` 目录下使用以下命令编译代码：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build
```

如果 `cmake` 命令无法找到 Eigen，可以在 `cmake` 命令后添加 `-DEigen3_DIR=/path/to/eigen` 参数指定 Eigen 的路径。大作业中的代码经过测试，可以在 ArchLinux 上使用 GCC 14.2.1 或 Clang 19 编译，使用其他编译器可能会出现问题，如果遇到问题，请检查编译器的版本。

### 运行代码

编译完成后，可以在 `build/` 目录下找到可执行文件 `Final`，可以使用 `./Final -h` 查看帮助信息。

## 在 GPU 上运行 Python 代码

### 运行环境

本次大作业中还使用了 Python 中的 [Triton](https://triton-lang.org/main/index.html) 库在 GPU 上实现了 V-cycle 多重网格方法。为了在 GPU 上运行代码，需要安装 Triton，安装方法可以参考 Triton 的官方文档，由于目前 Triton 仅支持 Linux 平台，因此需要在 Linux 上运行代码。Python 代码可以选择使用 [PyTorch](https://pytorch.org) 或 [JAX](https://github.com/jax-ml/jax) 作为向量库，请确保安装了 PyTorch 或 JAX 之一。如果使用 JAX，还需要安装 [jax-triton](https://github.com/jax-ml/jax-triton)。

经过测试，确认可以在以下环境中运行代码：

- `Python` 3.13.1
- `Triton` 3.2.0
- `PyTorch` 2.6.0 (Optional, for PyTorch backend)
- `JAX` 0.5.0 (Optional, for JAX backend)
- `jax-triton` 从 git 提交 `c3419d6555f2b64470788b02a27d11a44adb09ab` 构建 (Optional, for JAX backend)

### 运行代码

在 `python/` 目录下可以找到 Python 代码，可以使用以下命令运行 PyTorch 后端：

```bash
python3 main.py -b torch
```

或者使用以下命令运行 JAX 后端：

```bash
python3 main.py -b jax
```