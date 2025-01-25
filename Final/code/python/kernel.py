import triton
import triton.language as tl


###########################################
### Kernels for Gauss Seidel iterations ###
###########################################

@triton.jit
def gs_kernel_u_inner(u_ptr, fu_ptr, out_ptr, n: int, shift: int, block_size: tl.constexpr):
    # u: (n, n-1), fu: (n, n-1)
    pid = tl.program_id(axis=0)
    y = pid * block_size * 2 + shift + tl.arange(0, block_size) * 2
    offsets = y * (n - 1)
    factor = tl.where(y == 0 or y == n - 1, 3 * n * n, 4 * n * n)

    left = tl.zeros((block_size,), tl.float64)
    for i in tl.range(n - 1, loop_unroll_factor=2):
        base_offsets = offsets + i
        right = tl.load(u_ptr + base_offsets + 1) if i != n - 2 else tl.zeros((block_size,), tl.float64)
        bottom = tl.load(u_ptr + base_offsets - (n - 1), mask=y != 0, other=0.0)
        top = tl.load(u_ptr + base_offsets + (n - 1), mask=y != n - 1, other=0.0)
        fu = tl.load(fu_ptr + base_offsets)

        new_val = (fu + n * n * (left + right + top + bottom)) / factor
        tl.store(u_ptr + base_offsets, new_val)
        if out_ptr is not None:
            tl.store(out_ptr + base_offsets, new_val)
        left = new_val


@triton.jit
def gs_kernel_u_inplace(u_ptr, fu_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_u_inner(u_ptr, fu_ptr, None, n, 0, block_size)
    gs_kernel_u_inner(u_ptr, fu_ptr, None, n, 1, block_size)


@triton.jit
def gs_kernel_u_inplace_rev(u_ptr, fu_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_u_inner(u_ptr, fu_ptr, None, n, 1, block_size)
    gs_kernel_u_inner(u_ptr, fu_ptr, None, n, 0, block_size)


@triton.jit
def gs_kernel_u(u_ptr, fu_ptr, out_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_u_inner(u_ptr, fu_ptr, out_ptr, n, 0, block_size)
    gs_kernel_u_inner(u_ptr, fu_ptr, out_ptr, n, 1, block_size)


@triton.jit
def gs_kernel_u_rev(u_ptr, fu_ptr, out_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_u_inner(u_ptr, fu_ptr, out_ptr, n, 1, block_size)
    gs_kernel_u_inner(u_ptr, fu_ptr, out_ptr, n, 0, block_size)


@triton.jit
def gs_kernel_v_inner(v_ptr, fv_ptr, out_ptr, n: int, shift: int, block_size: tl.constexpr):
    # v: (n-1, n), fv: (n-1, n)
    pid = tl.program_id(axis=0)
    x = pid * block_size * 2 + shift + tl.arange(0, block_size) * 2
    factor = tl.where(x == 0 or x == n - 1, 3 * n * n, 4 * n * n)

    bottom = tl.zeros((block_size,), tl.float64)
    for i in tl.range(n - 1, loop_unroll_factor=2):
        base_offsets = x + i * n
        top = tl.load(v_ptr + base_offsets + n) if i != n - 2 else tl.zeros((block_size,), tl.float64)
        left = tl.load(v_ptr + base_offsets - 1, mask=x != 0, other=0.0)
        right = tl.load(v_ptr + base_offsets + 1, mask=x != n - 1, other=0.0)
        fv = tl.load(fv_ptr + base_offsets)

        new_val = (fv + n * n * (left + right + top + bottom)) / factor
        tl.store(v_ptr + base_offsets, new_val)
        if out_ptr is not None:
            tl.store(out_ptr + base_offsets, new_val)
        bottom = new_val


@triton.jit
def gs_kernel_v_inplace(v_ptr, fv_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_v_inner(v_ptr, fv_ptr, None, n, 0, block_size)
    gs_kernel_v_inner(v_ptr, fv_ptr, None, n, 1, block_size)


@triton.jit
def gs_kernel_v_inplace_rev(v_ptr, fv_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_v_inner(v_ptr, fv_ptr, None, n, 1, block_size)
    gs_kernel_v_inner(v_ptr, fv_ptr, None, n, 0, block_size)


@triton.jit
def gs_kernel_v(v_ptr, fv_ptr, out_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_v_inner(v_ptr, fv_ptr, out_ptr, n, 0, block_size)
    gs_kernel_v_inner(v_ptr, fv_ptr, out_ptr, n, 1, block_size)


@triton.jit
def gs_kernel_v_rev(v_ptr, fv_ptr, out_ptr, n: int, block_size: tl.constexpr):
    gs_kernel_v_inner(v_ptr, fv_ptr, out_ptr, n, 1, block_size)
    gs_kernel_v_inner(v_ptr, fv_ptr, out_ptr, n, 0, block_size)


#########################################
### Kernels for calculating rhs of GS ###
#########################################

@triton.jit
def cal_fu_kernel(fu_ptr, p_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    x = offsets % (n - 1)
    y = offsets // (n - 1)
    p_offsets_left = x + y * n
    p_offsets_right = x + 1 + y * n

    p_left = tl.load(p_ptr + p_offsets_left)
    p_right = tl.load(p_ptr + p_offsets_right)
    fu = tl.load(fu_ptr + offsets)

    out = fu - n * (p_right - p_left)
    tl.store(out_ptr + offsets, out)


@triton.jit
def cal_fv_kernel(fv_ptr, p_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    p_bottom = tl.load(p_ptr + offsets)
    p_top = tl.load(p_ptr + offsets + n)
    fv = tl.load(fv_ptr + offsets)

    out = fv - n * (p_top - p_bottom)
    tl.store(out_ptr + offsets, out)


##############################################
### Kernel for updating pressure and speed ###
##############################################

@triton.jit
def update_pressure_kernel_inner(u_ptr, v_ptr, p_ptr, d_ptr,
                                 out_u_ptr, out_v_ptr, out_p_ptr,
                                 n: int, shift: int, total_num: int,
                                 block_size: tl.constexpr):
    # p: (n, n), d: (n, n), u: (n, n-1), v: (n-1, n)
    pid = tl.program_id(axis=0)
    block_start = pid * block_size * total_num + shift
    x = block_start + tl.arange(0, block_size) * total_num

    v_bottom = tl.zeros((block_size,), tl.float64)
    p_bottom = tl.zeros((block_size,), tl.float64)
    p = tl.load(p_ptr + x)
    for i in tl.range(n, loop_unroll_factor=8):
        offsets = x + i * n
        y = tl.full((block_size,), i, dtype=tl.int32)
        d = tl.load(d_ptr + offsets)
        u_right = tl.load(u_ptr + x + y * (n - 1), mask=x != n - 1, other=0)
        u_left = tl.load(u_ptr + x - 1 + y * (n - 1), mask=x != 0, other=0)
        v_top = tl.load(v_ptr + x + y * n, mask=y != n - 1, other=0)
        r = -d - n * (u_right - u_left + v_top - v_bottom)

        left = tl.where(x != 0, 1.0, 0.0)
        right = tl.where(x != n - 1, 1.0, 0.0)
        top = tl.where(y != n - 1, 1.0, 0.0)
        bottom = tl.where(y != 0, 1.0, 0.0)
        cnt = left + right + top + bottom

        # update speed
        delta = r / cnt / n
        u_left -= delta
        u_right += delta
        v_bottom -= delta
        v_top += delta
        tl.store(u_ptr + x + y * (n - 1), u_right, mask=x != n - 1)
        tl.store(u_ptr + x - 1 + y * (n - 1), u_left, mask=x != 0)
        tl.store(v_ptr + x + (y - 1) * n, v_bottom, mask=y != 0)
        if out_u_ptr is not None:
            tl.store(out_u_ptr + x + y * (n - 1), u_right, mask=x != n - 1)
            tl.store(out_u_ptr + x - 1 + y * (n - 1), u_left, mask=x != 0)
            tl.store(out_v_ptr + x + (y - 1) * n, v_bottom, mask=y != 0)

        # update pressure
        p += r
        r = r / cnt
        p_left = tl.load(p_ptr + x - 1 + y * n, mask=x != 0)
        p_right = tl.load(p_ptr + x + 1 + y * n, mask=x != n - 1)
        p_top = tl.load(p_ptr + x + (y + 1) * n, mask=y != n - 1)
        p_left -= r
        p_right -= r
        p_bottom -= r
        p_top -= r
        tl.store(p_ptr + x - 1 + y * n, p_left, mask=x != 0)
        tl.store(p_ptr + x + 1 + y * n, p_right, mask=x != n - 1)
        tl.store(p_ptr + x + (y - 1) * n, p_bottom, mask=y != 0)
        if out_p_ptr is not None:
            tl.store(out_p_ptr + x - 1 + y * n, p_left, mask=x != 0)
            tl.store(out_p_ptr + x + 1 + y * n, p_right, mask=x != n - 1)
            tl.store(out_p_ptr + x + (y - 1) * n, p_bottom, mask=y != 0)

        v_bottom = v_top
        p_bottom = p
        p = p_top

    tl.store(p_ptr + x + n * (n - 1), p_bottom)
    if out_p_ptr is not None:
        tl.store(out_p_ptr + x + n * (n - 1), p_bottom)


@triton.jit
def update_pressure_kernel_inplace(u_ptr, v_ptr, p_ptr, d_ptr, n: int, total_num: int,
                                   block_size: tl.constexpr):
    for i in tl.range(total_num):
        update_pressure_kernel_inner(u_ptr, v_ptr, p_ptr, d_ptr, None, None, None, n, i, total_num, block_size)


@triton.jit
def update_pressure_kernel_inplace_rev(u_ptr, v_ptr, p_ptr, d_ptr, n: int, total_num: int,
                                       block_size: tl.constexpr):
    for i in tl.range(total_num - 1, -1, -1):
        update_pressure_kernel_inner(u_ptr, v_ptr, p_ptr, d_ptr, None, None, None, n, i, total_num, block_size)


@triton.jit
def update_pressure_kernel(u_ptr, v_ptr, p_ptr, d_ptr, out_u_ptr, out_v_ptr, out_p_ptr, n: int, total_num: int,
                           block_size: tl.constexpr):
    for i in tl.range(total_num):
        update_pressure_kernel_inner(u_ptr, v_ptr, p_ptr, d_ptr,
                                     out_u_ptr, out_v_ptr, out_p_ptr,
                                     n, i, total_num, block_size)


@triton.jit
def update_pressure_kernel_rev(u_ptr, v_ptr, p_ptr, d_ptr, out_u_ptr, out_v_ptr, out_p_ptr, n: int, total_num: int,
                               block_size: tl.constexpr):
    for i in tl.range(total_num - 1, -1, -1):
        update_pressure_kernel_inner(u_ptr, v_ptr, p_ptr, d_ptr,
                                     out_u_ptr, out_v_ptr, out_p_ptr,
                                     n, i, total_num, block_size)


######################################
### Kernels for calculating errors ###
######################################

@triton.jit
def error_u_kernel(u_ptr, p_ptr, fu_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    x = offsets % (n - 1)
    y = offsets // (n - 1)

    fu = tl.load(fu_ptr + offsets)
    p_left = tl.load(p_ptr + x + y * n)
    p_right = tl.load(p_ptr + x + 1 + y * n)
    u = tl.load(u_ptr + offsets)
    u_left = tl.load(u_ptr + offsets - 1, mask=x != 0, other=0.0)
    u_right = tl.load(u_ptr + offsets + 1, mask=x != n - 2, other=0.0)
    u_top = tl.load(u_ptr + offsets + n - 1, mask=y != n - 1, other=0.0)
    u_bottom = tl.load(u_ptr + offsets - (n - 1), mask=y != 0, other=0.0)

    cnt = tl.where(y != 0, 1.0, 0.0) + tl.where(y != n - 1, 1.0, 0.0)
    error = fu - n * (p_right - p_left - n * (u_right + u_left + u_bottom + u_top - (2 + cnt) * u))
    tl.store(out_ptr + offsets, error)


@triton.jit
def error_v_kernel(v_ptr, p_ptr, fv_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    x = offsets % n
    y = offsets // n

    fv = tl.load(fv_ptr + offsets)
    p_bottom = tl.load(p_ptr + x + y * n)
    p_top = tl.load(p_ptr + x + (y + 1) * n)
    v = tl.load(v_ptr + offsets)
    v_left = tl.load(v_ptr + offsets - 1, mask=x != 0, other=0.0)
    v_right = tl.load(v_ptr + offsets + 1, mask=x != n - 1, other=0.0)
    v_top = tl.load(v_ptr + offsets + n, mask=y != n - 2, other=0.0)
    v_bottom = tl.load(v_ptr + offsets - n, mask=y != 0, other=0.0)

    cnt = tl.where(x != 0, 1.0, 0.0) + tl.where(x != n - 1, 1.0, 0.0)
    error = fv - n * (p_top - p_bottom - n * (v_top + v_bottom + v_left + v_right - (2 + cnt) * v))
    tl.store(out_ptr + offsets, error)


@triton.jit
def error_p_kernel(u_ptr, v_ptr, d_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    x = offsets % n
    y = offsets // n

    d = tl.load(d_ptr + offsets)
    u_right = tl.load(u_ptr + x + y * (n - 1), mask=x != n - 1, other=0.0)
    u_left = tl.load(u_ptr + x - 1 + y * (n - 1), mask=x != 0, other=0.0)
    v_top = tl.load(v_ptr + offsets, mask=y != n - 1, other=0.0)
    v_bottom = tl.load(v_ptr + offsets - n, mask=y != 0, other=0.0)

    error = d + n * (u_right - u_left + v_top - v_bottom)
    tl.store(out_ptr + offsets, error)


###############################
### Kernels for restriction ###
###############################

@triton.jit
def restrict_u_kernel(u_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    n_new = n // 2
    x = offsets % (n_new - 1)
    y = offsets // (n_new - 1)

    bottom_offsets = 2 * x + 2 * y * (n - 1) + 1
    top_offsets = bottom_offsets + n - 1

    bottom = tl.load(u_ptr + bottom_offsets)
    top = tl.load(u_ptr + top_offsets)
    left_bottom = tl.load(u_ptr + bottom_offsets - 1)
    right_bottom = tl.load(u_ptr + bottom_offsets + 1)
    left_top = tl.load(u_ptr + top_offsets - 1)
    right_top = tl.load(u_ptr + top_offsets + 1)

    restrict_val = (top + bottom) / 4 + (left_bottom + right_bottom + left_top + right_top) / 8
    tl.store(out_ptr + offsets, restrict_val)


@triton.jit
def restrict_v_kernel(v_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    n_new = n // 2
    x = offsets % n_new
    y = offsets // n_new

    left_offsets = 2 * x + (2 * y + 1) * n
    right_offsets = left_offsets + 1

    left = tl.load(v_ptr + left_offsets)
    right = tl.load(v_ptr + right_offsets)
    left_bottom = tl.load(v_ptr + left_offsets - n)
    right_bottom = tl.load(v_ptr + right_offsets - n)
    left_top = tl.load(v_ptr + left_offsets + n)
    right_top = tl.load(v_ptr + right_offsets + n)

    restrict_val = (left + right) / 4 + (left_bottom + right_bottom + left_top + right_top) / 8
    tl.store(out_ptr + offsets, restrict_val)


@triton.jit
def restrict_p_kernel(p_ptr, out_ptr, n: int, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    n_new = n // 2
    x = offsets % n_new
    y = offsets // n_new

    bl_offsets = 2 * x + 2 * y * n

    bottom_left = tl.load(p_ptr + bl_offsets)
    bottom_right = tl.load(p_ptr + bl_offsets + 1)
    top_left = tl.load(p_ptr + bl_offsets + n)
    top_right = tl.load(p_ptr + bl_offsets + n + 1)

    restrict_val = (bottom_left + bottom_right + top_left + top_right) / 4
    tl.store(out_ptr + offsets, restrict_val)


###########################
### Kernels for lifting ###
###########################

@triton.jit
def lift_u_kernel(u_ptr, out_ptr, n: int, block_size: tl.constexpr):
    n_old = n // 2
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    x = offsets % (n - 1)
    y = offsets // (n - 1)
    x_old = x // 2
    y_old = y // 2

    nearest_offsets = x_old + y_old * (n_old - 1)
    nearest = tl.load(u_ptr + nearest_offsets, mask=x_old != n_old - 1, other=0.0)
    left = tl.load(u_ptr + nearest_offsets - 1, mask=x_old != 0, other=0.0)
    lift_val = tl.where(x % 2 == 0, (left + nearest) / 2, nearest)

    tl.store(out_ptr + offsets, lift_val)


@triton.jit
def lift_v_kernel(v_ptr, out_ptr, n: int, block_size: tl.constexpr):
    n_old = n // 2
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    x = offsets % n
    y = offsets // n
    x_old = x // 2
    y_old = y // 2

    nearest_offsets = x_old + y_old * n_old
    nearest = tl.load(v_ptr + nearest_offsets, mask=y_old != n_old - 1, other=0.0)
    bottom = tl.load(v_ptr + nearest_offsets - n_old, mask=y_old != 0, other=0.0)
    lift_val = tl.where(y % 2 == 0, (bottom + nearest) / 2, nearest)

    tl.store(out_ptr + offsets, lift_val)


@triton.jit
def lift_p_kernel(p_ptr, out_ptr, n: int, block_size: tl.constexpr):
    n_old = n // 2
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    x = offsets % n
    y = offsets // n
    y_old = y // 2
    x_old = x // 2

    nearest_offsets = x_old + y_old * n_old
    nearest = tl.load(p_ptr + nearest_offsets)
    tl.store(out_ptr + offsets, nearest)
