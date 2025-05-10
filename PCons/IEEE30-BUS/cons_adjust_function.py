import numpy as np
import pandas as pd
import warnings
import numba as nb
from numba import jit
import cvxpy as cp


# 辅助函数JIT加速
# @jit(nopython=True)
def calculate_marginal_cost(p_current, a, b):
    return 2 * a * p_current + b


def enforce_ramping_constraints(clamped_3d, status_3d, constraints):
    """向量化爬坡约束优化版本"""
    clamped = np.copy(clamped_3d)
    num_days, num_periods, num_units = clamped.shape

    # 向量化约束参数
    min_p = constraints['最小'].values.reshape(1, 1, -1)
    max_p = constraints['最大'].values.reshape(1, 1, -1)
    ramp_up = constraints['爬坡'].values.reshape(1, 1, -1)
    ramp_down = constraints['滑坡'].values.reshape(1, 1, -1)

    # 批量处理天维度
    prev_power = clamped[:, 0, :].copy()  # 初始化为首时段数据

    for t in range(1, num_periods):
        current_status = status_3d[:, t, :]

        # 计算动态约束
        ramp_min = prev_power - ramp_down
        ramp_max = prev_power + ramp_up
        final_min = np.maximum(min_p, ramp_min)
        final_max = np.minimum(max_p, ramp_max)

        # 批量截断
        current_power = np.clip(clamped[:, t, :], final_min, final_max)

        # 更新功率并考虑停机状态
        clamped[:, t, :] = np.where(current_status == 1, current_power, 0)
        prev_power = np.where(current_status == 1, current_power, 0)

    return clamped


def multi_segment_clamp(predictions_3d, constraints):
    """全向量化多段式截断"""
    min_p = constraints['最小'].values.reshape(1, 1, -1)
    max_p = constraints['最大'].values.reshape(1, 1, -1)
    threshold = min_p / 3

    return np.select(
        condlist=[
            predictions_3d < 0,
            (predictions_3d >= 0) & (predictions_3d < threshold),
            (predictions_3d >= threshold) & (predictions_3d < min_p),
            predictions_3d > max_p
        ],
        choicelist=[0, 0, min_p, max_p],
        default=predictions_3d
    )


def enforce_status_power_consistency(clamped_3d, status_3d, constraints):
    """向量化状态一致性处理"""
    min_p = constraints['最小'].values.reshape(1, 1, -1)
    max_p = constraints['最大'].values.reshape(1, 1, -1)

    clamped_3d = np.where(status_3d == 1,
                          np.clip(clamped_3d, min_p, max_p),
                          0)
    return clamped_3d, status_3d


def allocate_power_deficit(clamped_3d, status_3d, constraints, demand_profile):
    """供电缺口分配函数（优化版：按天分组+向量化）"""
    adjusted = np.copy(clamped_3d)
    num_days, num_periods, num_units = clamped_3d.shape

    # 预计算机组参数矩阵（加速访问）
    ramp_up = constraints['爬坡'].values.reshape(1, -1)  # (1, num_units)
    ramp_down = constraints['滑坡'].values.reshape(1, -1)
    max_p = constraints['最大'].values.reshape(1, -1)
    min_p = constraints['最小'].values.reshape(1, -1)
    a = constraints['a'].values.reshape(1, -1)
    b = constraints['b'].values.reshape(1, -1)

    for period in range(num_periods):
        # 当前时段数据（保留三维结构）
        current_power = adjusted[:, period, :]  # (num_days, num_units)
        current_status = status_3d[:, period, :]
        prev_power = adjusted[:, period - 1, :] if period > 0 else current_power
        next_power = adjusted[:, period + 1, :] if period < num_periods - 1 else current_power

        # 计算各天缺口（向量化）
        total_gen = np.sum(current_power * current_status, axis=1)  # (num_days,)
        deficits = demand_profile[:, period] - total_gen  # (num_days,)
        needs_processing = np.where(deficits > 1e-6)[0]

        for day in needs_processing:
            # 提取单天数据
            day_status = current_status[day]  # (num_units,)
            active_units = np.where(day_status == 1)[0]
            if len(active_units) == 0:
                raise ValueError(f"Day {day} Period {period}: 无可用机组")

            # 向量化计算调节能力（关键优化）
            prev_p = prev_power[day, active_units]
            next_p = next_power[day, active_units] if period < num_periods - 1 else None

            # 动态约束计算（向量化）
            upper_bounds = np.minimum(
                max_p[0, active_units],
                prev_p + ramp_up[0, active_units]
            )
            if next_p is not None:
                next_lower = np.maximum(
                    current_power[day, active_units] - ramp_down[0, active_units],
                    min_p[0, active_units]
                )
                dynamic_upper = np.minimum(
                    next_p - next_lower + current_power[day, active_units],
                    max_p[0, active_units]
                )
                upper_bounds = np.minimum(upper_bounds, dynamic_upper)

            headrooms = np.maximum(0, upper_bounds - current_power[day, active_units])
            valid_units = np.where(headrooms > 0)[0]
            adjust_capacity = list(zip(active_units[valid_units], headrooms[valid_units]))

            # 边际成本排序
            sorted_units = sorted(
                adjust_capacity,
                key=lambda x: calculate_marginal_cost(
                    current_power[day, x[0]],
                    a[0, x[0]],
                    b[0, x[0]]
                )
            )

            # 分配缺口
            remaining = deficits[day]
            for unit_idx, capacity in sorted_units:
                allocate = min(remaining, capacity)
                adjusted[day, period, unit_idx] += allocate
                remaining -= allocate
                if remaining <= 1e-6:
                    break

            if remaining > 1e-6:
                warnings.warn(f"Day {day} Period {period} 剩余缺口: {remaining:.2f}MW")

    return adjusted


def allocate_power_surplus(clamped_3d, status_3d, constraints, demand_profile):
    """优化版供电超额调整函数（向量化+按天处理）"""
    adjusted = np.copy(clamped_3d)
    num_days, num_periods, num_units = clamped_3d.shape

    # 预计算约束参数矩阵（加速访问）
    min_p = constraints['最小'].values.reshape(1, -1)  # (1, num_units)
    max_p = constraints['最大'].values.reshape(1, -1)
    ramp_up = constraints['爬坡'].values.reshape(1, -1)
    ramp_down = constraints['滑坡'].values.reshape(1, -1)
    a = constraints['a'].values.reshape(1, -1)
    b = constraints['b'].values.reshape(1, -1)

    for period in range(num_periods):
        # 当前时段数据（保持三维结构）
        current_power = adjusted[:, period, :]  # (num_days, num_units)
        current_status = status_3d[:, period, :]
        prev_power = adjusted[:, period - 1, :] if period > 0 else current_power
        next_power = adjusted[:, period + 1, :] if period < num_periods - 1 else current_power

        # 计算各天超额量（向量化）
        total_gen = np.sum(current_power * current_status, axis=1)  # (num_days,)
        surpluses = total_gen - demand_profile[:, period]  # (num_days,)
        needs_processing = np.where(surpluses > 1e-6)[0]

        for day in needs_processing:
            # 提取单天数据
            day_power = current_power[day]  # (num_units,)
            day_status = current_status[day]
            active_units = np.where(day_status == 1)[0]
            if len(active_units) == 0:
                raise ValueError(f"Day {day} Period {period}: 无可用机组")

            # 向量化计算调节能力（关键优化）
            prev_p = prev_power[day, active_units]
            next_p = next_power[day, active_units] if period < num_periods - 1 else None

            # 动态下界计算（向量化）
            lower_bounds = np.maximum(
                min_p[0, active_units],
                prev_p - ramp_down[0, active_units]
            )
            if next_p is not None:
                next_upper = np.minimum(
                    current_power[day, active_units] + ramp_up[0, active_units],
                    max_p[0, active_units]
                )
                dynamic_upper = np.maximum(
                    current_power[day, active_units] - next_upper + next_p,
                    min_p[0, active_units]
                )
                lower_bounds = np.maximum(lower_bounds, dynamic_upper)

            headrooms = np.maximum(0, day_power[active_units] - lower_bounds)
            valid_units = np.where(headrooms > 0)[0]
            adjust_capacity = list(zip(active_units[valid_units], headrooms[valid_units]))

            # 边际成本排序（降序）
            sorted_units = sorted(
                adjust_capacity,
                key=lambda x: -calculate_marginal_cost(
                    day_power[x[0]],
                    a[0, x[0]],
                    b[0, x[0]]
                )
            )

            # 分配超额量
            remaining = surpluses[day]
            for unit_idx, capacity in sorted_units:
                allocate = min(remaining, capacity)
                adjusted[day, period, unit_idx] -= allocate
                remaining -= allocate
                if remaining <= 1e-6:
                    break

            if remaining > 1e-6:
                warnings.warn(f"Day {day} Period {period} 未消纳超额: {remaining:.2f}MW")

    return adjusted


def calculate_upper_bounds(prev_power, current_power, next_power,
                           max_p, min_p, ramp_up, ramp_down,
                           active_units):
    """
    综合上界计算函数（向量化版本）

    参数:
        prev_power      : 前一时刻出力向量 (num_units,)
        current_power   : 当前时刻出力向量 (num_units,)
        next_power      : 下一时刻出力向量 (num_units,) 或 None
        max_p           : 最大出力约束向量 (num_units,)
        min_p           : 最小出力约束向量 (num_units,)
        ramp_up         : 爬坡率约束向量 (num_units,)
        ramp_down       : 滑坡率约束向量 (num_units,)
        active_units    : 当前可调机组索引数组 (例如 np.where(status==1)[0])

    返回:
        upper_bounds    : 动态上界向量 (len(active_units),)
    """
    # 提取活跃机组数据
    active_prev = prev_power[active_units]
    active_max = max_p[active_units]
    active_ramp_up = ramp_up[active_units]

    # 初始上界计算
    upper_bounds = np.minimum(active_max, active_prev + active_ramp_up)

    # 动态约束处理（如果存在下一时刻出力）
    if next_power is not None:
        active_current = current_power[active_units]
        active_next = next_power[active_units]
        active_min = min_p[active_units]
        active_ramp_down = ramp_down[active_units]

        # 计算下一时刻下限约束
        next_lower = np.maximum(
            active_current - active_ramp_down,
            active_min
        )

        # 计算动态上界调整量
        dynamic_upper = np.minimum(
            active_next - next_lower + active_current,
            active_max
        )

        # 综合上界
        upper_bounds = np.minimum(upper_bounds, dynamic_upper)

    return upper_bounds


def calculate_lower_bounds(prev_power, current_power, next_power,
                           min_p, max_p, ramp_down, ramp_up,
                           active_units):
    """
    动态下界计算函数（严格保持原始逻辑）

    参数:
        prev_power      : 前一时段出力 (num_units,)
        current_power   : 当前时段出力 (num_units,)
        next_power      : 下一时段出力 (num_units,) 或 None
        min_p           : 最小出力约束 (num_units,)
        max_p           : 最大出力约束 (num_units,)
        ramp_down       : 滑坡率约束 (num_units,)
        ramp_up         : 爬坡率约束 (num_units,)
        active_units    : 活跃机组索引数组

    返回:
        lower_bounds    : 动态下界向量 (len(active_units),)
    """
    # 提取活跃机组参数
    active_prev = prev_power[active_units]
    active_min = min_p[active_units]
    active_ramp_down = ramp_down[active_units]
    active_current = current_power[active_units]

    # 初始下界计算
    lower_bounds = np.maximum(active_min, active_prev - active_ramp_down)

    # 动态约束处理（存在下一时段时）
    if next_power is not None:
        active_next = next_power[active_units]
        active_max = max_p[active_units]
        active_ramp_up = ramp_up[active_units]

        # 计算下一时段允许的最大出力
        next_upper = np.minimum(
            active_current + active_ramp_up,
            active_max
        )

        # 计算动态约束调整量
        dynamic_upper = np.maximum(
            active_current - next_upper + active_next,
            active_min
        )

        # 合并下界约束
        lower_bounds = np.maximum(lower_bounds, dynamic_upper)

    return lower_bounds


def build_optimization_model(reducible_units, transfer_units,
                             max_decrease, max_transfer,
                             gen_ptdf, original_flow, line_limits,
                             a, b, current_power):
    """安全构建优化模型避免链式约束错误"""
    # 定义优化变量
    delta_reduce = cp.Variable(len(reducible_units), nonneg=True)  # 非负变量
    delta_transfer = cp.Variable(len(transfer_units), nonneg=True)

    # 构建约束列表
    constraints = []

    # 功率平衡约束
    constraints.append(cp.sum(delta_reduce) == cp.sum(delta_transfer))

    # 削减量上限约束（分离不等式）
    if len(reducible_units) > 0:
        constraints.append(delta_reduce <= max_decrease)

    # 转入量上限约束（分离不等式）
    if len(transfer_units) > 0:
        constraints.append(delta_transfer <= max_transfer)

    # 潮流约束重构
    if gen_ptdf is not None and original_flow is not None:
        # 构建调整影响矩阵
        adj_matrix = np.zeros((gen_ptdf.shape[0], len(reducible_units) + len(transfer_units)))
        adj_matrix[:, :len(reducible_units)] = -gen_ptdf[:, reducible_units]  # 削减影响
        adj_matrix[:, len(reducible_units):] = gen_ptdf[:, transfer_units]  # 转入影响

        # 计算总潮流变化
        total_flow_change = adj_matrix @ cp.hstack([delta_reduce, delta_transfer])
        corrected_flow = original_flow + total_flow_change

        # 添加绝对值约束（分解为两个不等式）
        constraints += [
            corrected_flow <= line_limits
        ]

    # 构建经济性目标函数
    cost_reduce = (2 * a[reducible_units] * current_power[reducible_units]
                   + b[reducible_units]) @ delta_reduce
    cost_transfer = (2 * a[transfer_units] * current_power[transfer_units]
                     + b[transfer_units]) @ delta_transfer
    objective = cp.Minimize(- cost_reduce + cost_transfer)  # 削减成本+转入收益

    return cp.Problem(objective, constraints), delta_reduce, delta_transfer


def enforce_ptdf_constraints(clamped_3d, status_3d, constraints,
                             gen_ptdf, load_ptdf, load_profile, line_limits,
                             max_iter=1):
    """
    完整PTDF约束修正函数（集成动态三界限）

    参数:
        clamped_3d    : (days, periods, units) 发电计划三维数组
        status_3d     : (days, periods, units) 机组状态三维数组
        constraints   : DataFrame 机组约束参数
        gen_ptdf      : (lines, units) 机组PTDF矩阵
        load_ptdf     : (lines, nodes) 负荷PTDF矩阵
        load_profile  : (days, periods, nodes) 节点负荷三维数组
        line_limits   : (lines,) 线路容量限制
        max_iter      : 最大迭代次数

    返回:
        adjusted      : 修正后的发电计划三维数组
    """
    # 参数预处理
    adjusted = np.copy(clamped_3d)
    num_days, num_periods, num_units = adjusted.shape
    num_lines = gen_ptdf.shape[0]

    # 约束参数矩阵化
    min_p = constraints['最小'].values
    max_p = constraints['最大'].values
    ramp_up = constraints['爬坡'].values
    ramp_down = constraints['滑坡'].values
    a = constraints['a'].values
    b = constraints['b'].values

    for _ in range(max_iter):
        # 计算当前潮流
        generation_flows = np.einsum('lu,dtu->dtl', gen_ptdf, adjusted)
        load_flows = np.einsum('lu,dtu->dtl', load_ptdf, load_profile)
        current_flows = np.abs(generation_flows - load_flows)

        # 检测需要处理的天和时段（按天+时段分组）
        violation_days_periods = np.argwhere(
            (current_flows > line_limits.reshape(1, 1, -1)).any(axis=2)
        )
        if not len(violation_days_periods):
            break

        # 按最大越限量排序处理顺序
        sorted_points = sorted(violation_days_periods,
                               key=lambda x: -np.max(current_flows[x[0], x[1]]))

        for day, period in sorted_points:
            current_power = adjusted[day, period].copy()
            status = status_3d[day, period]
            active_units = np.where(status == 1)[0]

            # ====== 动态约束计算 ======
            prev_p = adjusted[day, period - 1] if period > 0 else current_power
            next_p = adjusted[day, period + 1] if period < num_periods - 1 else None

            # 计算上下界
            lower_bounds = calculate_lower_bounds(
                prev_p, current_power, next_p,
                min_p, max_p, ramp_down, ramp_up,
                active_units
            )
            upper_bounds = calculate_upper_bounds(
                prev_p, current_power, next_p,
                max_p, min_p, ramp_up, ramp_down,
                active_units
            )

            # 计算可削减范围
            current_active = current_power[active_units]
            max_decrease = current_active - lower_bounds  # 可下调量
            valid_decrease = max_decrease > 1e-6  # 有效调整标识
            # 筛选可调机组
            reducible_units = active_units[valid_decrease]
            if len(reducible_units) == 0:
                continue  # 无可削减机组

            # 计算可转入范围
            max_transfer = upper_bounds - current_active  # 可下调量
            valid_transfer = max_transfer > 1e-6  # 有效调整标识
            # 筛选可调机组
            transfer_units = active_units[valid_transfer]
            if len(transfer_units) == 0:
                continue  # 无可转入机组

            # 定义优化变量
            problem, delta_reduce, delta_transfer = build_optimization_model(
                reducible_units=reducible_units,
                transfer_units=transfer_units,
                max_decrease=max_decrease[valid_decrease],
                max_transfer=max_transfer[valid_transfer],
                gen_ptdf=gen_ptdf,
                original_flow=generation_flows[day, period] - load_flows[day, period],
                line_limits=line_limits,
                a=a,
                b=b,
                current_power=current_power
            )

            # 求解优化问题
            try:
                problem.solve(solver=cp.ECOS, verbose=False, max_iters=50)
            except cp.SolverError:
                try:
                    problem.solve(solver=cp.SCS, verbose=False)
                except Exception as e:
                    print(f"求解失败: {str(e)}")
                    continue

            # 应用优化结果
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # print(f"Day {day} Period {period}: 存在可行解，已调整，状态码 {problem.status}")
                # 更新机组出力
                adjusted[day, period, reducible_units] -= delta_reduce.value
                adjusted[day, period, transfer_units] += delta_transfer.value
            # else:
            #     print(f"Day {day} Period {period}: 无可行解，状态码 {problem.status}")
    return adjusted