# 如何运行代码

- 安装 [`Eigen`](https://eigen.tuxfamily.org/index.php?title=Main_Page) 线性代数库。对于 Linux 系统，可以使用系统的包管理器安装。对于 Windows 系统，可以使用 [`vcpkg`](https://vcpkg.io/en/) 安装或者直接下载源码。
- 配置 CMake 构建脚本。在代码目录执行以下命令：
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  ```

  如果 CMake 提示找不到 `Eigen` 库，可以使用以下命令：
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=</path/to/eigen>
  ```
  如果你使用 `vcpkg` 安装了 `Eigen` 库，也可以使用以下命令：
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=</path/to/vcpkg>/scripts/buildsystems/vcpkg.cmake
  ```
- 编译代码。在代码目录执行以下命令：
  ```bash
  cmake --build build
  ```