import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 16

from pypower.api import case118
from adjust_10000_OnOff import adjust_10000_unit_operations
from cons_adjust_function import enforce_ramping_constraints, multi_segment_clamp, enforce_status_power_consistency, \
    allocate_power_deficit, allocate_power_surplus, enforce_ptdf_constraints

# 参数配置
DAYS = 30
PERIODS_PER_DAY = 288  # 5分钟间隔
NUM_UNITS = 54

model = 'GATCN'  # 新增模型标识


def load_data(model):
    """数据加载函数（包含需求重构）"""
    # 读取完整预测数据（路径包含模型标识）
    input_dir = os.path.join('prediction', model)
    full_data = pd.read_csv(os.path.join(input_dir, 'test_predictions.csv'))
    predictions = full_data['Test_Predictions'].values

    # 重构需求数据
    full_data['day'] = np.arange(len(full_data)) // (PERIODS_PER_DAY * NUM_UNITS)
    full_data['period'] = (np.arange(len(full_data)) // NUM_UNITS) % PERIODS_PER_DAY

    demand_df = full_data.groupby(['day', 'period'])['Test_Targets'].sum().reset_index()
    demand_2d = demand_df.pivot(index='day', columns='period', values='Test_Targets').values

    # 验证数据维度
    assert predictions.shape[0] == DAYS * PERIODS_PER_DAY * NUM_UNITS, "预测数据量不匹配"
    assert demand_2d.shape == (DAYS, PERIODS_PER_DAY), "需求数据重构失败"

    # 读取约束文件
    constraints = pd.read_csv('data/IEEE118_constraints.csv', encoding='gb18030')
    assert len(constraints) == NUM_UNITS, "约束文件机组数不匹配"

    # 时间单位转换
    constraints['min_start_periods'] = constraints['最小运行时间'].astype(int)
    constraints['min_end_periods'] = constraints['最小停机时间'].astype(int)

    return (predictions.reshape(DAYS, PERIODS_PER_DAY, NUM_UNITS),
            demand_2d,
            constraints)


def load_ptdf_data():
    """PTDF数据加载（与主数据加载解耦）"""
    ptdf = pd.read_csv('data/ptdf_118.csv', header=None).values

    # 获取电网拓扑信息
    case_data = case118()
    gen_bus_indices = case_data["gen"][:, 0].astype(int) - 1  # 0-based索引
    load_bus_indices = np.where(case_data["bus"][:, 2] > 0)[0]

    # 分割PTDF矩阵
    gen_ptdf = ptdf[:, gen_bus_indices]
    load_ptdf = ptdf[:, load_bus_indices]

    # 加载负荷数据
    df = pd.read_csv("data/all_busloads_118.csv", index_col=None)
    reshaped_data = df.values.ravel(order='F').reshape((366, 99, 288)).transpose((0, 2, 1))
    test_indices = [194, 326, 65, 138, 34, 356, 161, 77, 73, 20, 99, 357, 302, 363, 63, 103, 97, 246, 116, 121, 227, 29,
                    255, 11, 88, 280, 107, 167, 31, 207]
    load_profile = reshaped_data[test_indices, :, :]

    # 线路容量
    line_limits = case_data["branch"][:, 5].astype(float)

    # 维度验证
    assert gen_ptdf.shape[1] == NUM_UNITS, f"机组PTDF列数应为{NUM_UNITS}"
    assert load_ptdf.shape[1] == load_profile.shape[
        2], f"负荷节点数不匹配 {load_ptdf.shape[1]} vs {load_profile.shape[2]}"

    return gen_ptdf, load_ptdf, load_profile, line_limits


def validate_generation(power_3d, status_3d, demand_2d, output_dir):
    """供电缺口/超额校验函数"""
    total_generation = np.sum(power_3d * status_3d, axis=2)
    deficit = demand_2d - total_generation

    # 分解缺口和超额
    deficit_positive = np.where(deficit > 0, deficit, 0)  # 供电缺口
    surplus_positive = np.where(deficit < 0, -deficit, 0)  # 供电超额（取正值）

    print("\n=== 供电平衡分析 ===")
    print(f"[缺口统计] 最大剩余缺口: {np.max(deficit_positive):.4f}MW")
    print(f"[缺口统计] 平均未满足率: {np.mean(deficit_positive[deficit > 0]) / np.mean(demand_2d):.2%}")

    print(f"\n[超额统计] 最大供电超额: {np.max(surplus_positive):.4f}MW")
    print(f"[超额统计] 平均超额率: {np.mean(surplus_positive[surplus_positive > 0]) / np.mean(demand_2d):.2%}")

    # 可视化典型日（同时显示缺口和超额）
    plt.figure(figsize=(12, 6))
    plt.plot(demand_2d[0], label='系统需求', color='#1f77b4', lw=2)
    plt.plot(total_generation[0], label='实际发电', color='#ff7f0e', lw=2, linestyle='--')

    # 缺口填充（需求>发电）
    plt.fill_between(
        range(len(deficit[0])),
        demand_2d[0],
        total_generation[0],
        where=(deficit[0] > 0),
        color='red', alpha=0.3, label='供电缺口'
    )

    # 超额填充（发电>需求）
    plt.fill_between(
        range(len(deficit[0])),
        demand_2d[0],
        total_generation[0],
        where=(deficit[0] < 0),
        color='green', alpha=0.3, label='供电超额'
    )

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title("典型日供需平衡分析（第0天）")
    plt.xlabel("时段（5分钟间隔）")
    plt.ylabel("功率（MW）")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'generation_balance.png'), bbox_inches='tight')
    plt.close()


def validate_ptdf_constraints(clamped_3d, gen_ptdf, load_ptdf, load_profile, line_limits, output_dir):
    """全维度潮流约束验证

    参数说明：
    clamped_3d: (days, periods, units) 调整后的机组出力
    gen_ptdf: (lines, units) 机组PTDF矩阵
    load_ptdf: (lines, load_nodes) 负荷PTDF矩阵
    load_profile: (days, periods, load_nodes) 负荷数据
    line_limits: (lines,) 线路容量限制
    """
    gen_flows = np.einsum('lu,dtu->dtl', gen_ptdf, clamped_3d)
    load_flows = np.einsum('lu,dtu->dtl', load_ptdf, load_profile)

    # 总潮流计算
    total_flows = np.abs(gen_flows - load_flows)  # (days, periods, lines)

    # 越限检测
    violation_mask = total_flows > line_limits.reshape(1, 1, -1)
    violation_count = np.sum(violation_mask)
    total_points = np.prod(total_flows.shape)

    # 统计指标
    max_violation = np.max(total_flows - line_limits.reshape(1, 1, -1), initial=0)
    avg_violation = np.mean(np.where(violation_mask, total_flows - line_limits.reshape(1, 1, -1), 0))

    print("\n=== 潮流约束验证结果 ===")
    print(f"线路总数: {len(line_limits)}")
    print(f"总检测点数: {total_points} (days×periods×lines)")
    print(f"越限点数: {violation_count}")
    print(f"越限点比例: {violation_count / total_points:.2%}")
    print(f"最大越限量: {max_violation:.2f}MW")
    print(f"平均越限量: {avg_violation:.2f}MW")

    # 可视化最严重线路
    line_violations = np.sum(violation_mask, axis=(0, 1))  # 各线路总违规次数
    worst_line = np.argmax(line_violations)
    plt.figure(figsize=(10, 6))
    plt.plot(total_flows[:, :, worst_line].flatten(), '.', alpha=0.6)
    plt.axhline(y=line_limits[worst_line], color='r', linestyle='--',
                label=f'线路{worst_line}容量限制')
    plt.title(f"最严重线路 {worst_line} 潮流分布")
    plt.ylabel("潮流值 (MW)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'worst_line_flow.png'))

    # 保存详细违规记录
    violation_records = []
    days, periods, lines = np.where(violation_mask)
    for d, t, l in zip(days, periods, lines):
        violation_records.append({
            'day': d,
            'period': t,
            'line': l,
            'flow_value': total_flows[d, t, l],
            'limit': line_limits[l]
        })
    pd.DataFrame(violation_records).to_csv(
        os.path.join(output_dir, 'ptdf_violations_detail.csv'), index=False)


def save_results(clamped_3d, status_3d, output_dir):
    """结果保存函数"""
    days_arr, periods_arr, units_arr = np.indices(clamped_3d.shape)
    result_df = pd.DataFrame({
        'day': days_arr.ravel(),
        'period': periods_arr.ravel(),
        'unit': units_arr.ravel(),
        'clamped_power': clamped_3d.ravel(),
        'status': status_3d.ravel()
    })
    result_df.to_csv(os.path.join(output_dir, 'final_dispatch_status.csv'), index=False)
    print(f"结果已保存至：{output_dir}")


def process_before_adjustment(pred_3d, demand_2d, constraints, output_dir):
    """修正前处理（仅基础截断）"""
    clamped_3d = pred_3d.copy()
    status_3d = (clamped_3d > 0).astype(int)
    validate_generation(clamped_3d, status_3d, demand_2d, output_dir)
    save_results(clamped_3d, status_3d, output_dir)
    return clamped_3d, status_3d


def process_after_adjustment(predictions_3d, demand_data, constraints,
                            gen_ptdf, load_ptdf, load_profile, line_limits, output_dir):
    """修正后处理逻辑（包含完整约束调整）"""
    # 初始化
    # 开始计时
    stat_time = time.time()
    # 第一阶段：基础功率截断
    print('第一阶段：基础功率截断')
    clamped_3d = multi_segment_clamp(predictions_3d, constraints)

    # 第二阶段：生成初始状态
    print('第二阶段：生成初始状态')
    status_3d = (clamped_3d > 0).astype(int)

    # 第三阶段：启停时间约束调整
    print('第三阶段：启停时间约束调整')
    min_start = constraints['min_start_periods'].values
    min_end = constraints['min_end_periods'].values

    for day in range(DAYS):
        # print(day)
        # 提取单日状态数据 (288, 6)
        daily_status = status_3d[day, :, :].copy()

        # 执行启停调整（包含跨时段约束校验）
        adjusted_status = adjust_10000_unit_operations(
            daily_status,
            min_start=min_start,
            min_end=min_end,
            for_flag=3  # 双重校验模式
        )

        # 回写调整结果
        status_3d[day, :, :] = adjusted_status

    # 第四阶段：状态-功率一致性处理
    print('第四阶段：状态-功率一致性处理')
    clamped_3d, status_3d = enforce_status_power_consistency(clamped_3d, status_3d, constraints)

    # 考虑最小功率较小，重新截断
    status_3d = (clamped_3d > 0).astype(int)

    # +++ 新增第五阶段：爬坡率约束修正 +++
    print('第五阶段：爬坡率约束修正')
    clamped_3d = enforce_ramping_constraints(clamped_3d, status_3d, constraints)

    # +++ 新增第六阶段：供电缺口分配 +++
    print('第六阶段：供电缺口分配')
    clamped_3d = allocate_power_deficit(
        clamped_3d=clamped_3d,
        status_3d=status_3d,
        constraints=constraints,
        demand_profile=demand_data
    )

    # 新增第七阶段：供电超额调整
    print('第七阶段：供电超额调整')
    clamped_3d = allocate_power_surplus(
        clamped_3d=clamped_3d,
        status_3d=status_3d,
        constraints=constraints,
        demand_profile=demand_data
    )

    print('第八阶段：PTDF潮流约束修正')
    clamped_3d = enforce_ptdf_constraints(
        clamped_3d=clamped_3d,
        status_3d=status_3d,
        constraints=constraints,
        gen_ptdf=gen_ptdf,
        load_ptdf=load_ptdf,
        load_profile=load_profile,
        line_limits=line_limits
    )
    end_time = time.time()
    all_time = end_time - stat_time
    print("all_time:", all_time)

    validate_generation(clamped_3d, status_3d, demand_data, output_dir)
    validate_ptdf_constraints(clamped_3d, gen_ptdf, load_ptdf, load_profile, line_limits, output_dir)
    save_results(clamped_3d, status_3d, output_dir)
    return clamped_3d, status_3d


def main():
    # 创建输出目录
    output_before = os.path.join('out', model, 'before_adjustment')
    output_after = os.path.join('out', model, 'after_adjustment')
    os.makedirs(output_before, exist_ok=True)
    os.makedirs(output_after, exist_ok=True)

    # 数据加载
    pred_3d, demand_2d, constraints = load_data(model)
    gen_ptdf, load_ptdf, load_profile, line_limits = load_ptdf_data()

    # 分阶段处理
    print("\n=== 修正前处理 ===")
    process_before_adjustment(pred_3d, demand_2d, constraints, output_before)

    print("\n=== 修正后处理 ===")
    process_after_adjustment(pred_3d, demand_2d, constraints, gen_ptdf, load_ptdf, load_profile, line_limits,
                             output_after)


if __name__ == "__main__":
    main()