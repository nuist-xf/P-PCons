import numpy as np

def adjust_10000_unit_operations(unit_predict, min_start, min_end, for_flag=2):
    def compress_stages(tg, stage):
        if not tg or not stage or len(tg) != len(stage):
            return tg, stage  # 返回原始列表，如果输入不合规
        new_tg = []
        new_stage = []
        current_tg_sum = tg[0]
        current_stage = stage[0]
        for i in range(1, len(stage)):
            if stage[i] == current_stage:  # 相邻相同
                current_tg_sum += tg[i]  # 累加时长
            else:
                new_tg.append(current_tg_sum)  # 保存累加的时长
                new_stage.append(current_stage)  # 保存当前状态
                current_stage = stage[i]  # 更新当前状态
                current_tg_sum = tg[i]  # 重置累加时长
        # 添加最后一个阶段
        new_tg.append(current_tg_sum)
        new_stage.append(current_stage)
        return new_tg, new_stage
    for flag in range(for_flag):
        T, N = unit_predict.shape
        for g in range(N):
            change_flag = 0
            last_Ug = unit_predict[0, g]
            for t in range(1, T):
                if unit_predict[t, g] != last_Ug:
                    change_flag = 1
                    last_Ug = unit_predict[t, g]

            if change_flag == 1:
                tg = []
                stage = []
                current_stage = unit_predict[0, g]
                start_time = 0
                for t in range(1, T):
                    if unit_predict[t, g] != current_stage:
                        tg.append(t - start_time)
                        stage.append(current_stage)
                        start_time = t
                        current_stage = unit_predict[t, g]
                tg.append(T - start_time)
                stage.append(current_stage)

                # 新增的逻辑
                if min_start[g] > 288:
                    has_change = any(unit_predict[t, g] != unit_predict[0, g] for t in range(T))
                    if has_change:
                        unit_predict[:, g] = 1  # 将整个机组设置为启动
                    continue  # 跳过后续调整逻辑

                i = 0
                while i < len(tg):
                    if len(tg) <= 1:
                        break
                    adjusted = False
                    if i == 0 and ((stage[i] == 1 and tg[i] < min_start[g] and tg[i] < tg[i + 1]) or
                                   (stage[i] == 0 and tg[i] < min_end[g] and tg[i] < tg[i + 1])):
                        start = 0
                        end = tg[i]
                        unit_predict[start:end, g] = stage[i + 1]
                        adjusted = True
                    elif 0 < i < len(tg) - 1 and (
                            (stage[i] == 1 and tg[i] < min_start[g] and (tg[i] <= tg[i - 1] or tg[i] <= tg[i + 1])) or
                            (stage[i] == 0 and tg[i] < min_end[g] and (tg[i] <= tg[i - 1] or tg[i] <= tg[i + 1]))):
                        start = sum(tg[:i])
                        end = start + tg[i]
                        unit_predict[start:end, g] = stage[i + 1]
                        adjusted = True
                    elif flag == (for_flag - 1):
                        if 0 < i < len(tg) - 1 and (
                                (stage[i] == 1 and tg[i] < min_start[g] and ((
                                        np.abs(min_start[g] - tg[i]) >= np.abs(tg[i - 1] - min_end[g])) or (
                                        np.abs(min_start[g] - tg[i]) >= np.abs(tg[i + 1] - min_end[g]))))):
                            start = sum(tg[:i])
                            end = start + tg[i]
                            unit_predict[start:end, g] = stage[i + 1]
                            adjusted = True
                        if 0 < i < len(tg) - 1 and (
                                (stage[i] == 1 and tg[i] < min_start[g] and ((
                                        np.abs(min_start[g] - tg[i]) <= np.abs(tg[i - 1] - min_end[g])) or (
                                        np.abs(min_start[g] - tg[i]) <= np.abs(tg[i + 1] - min_end[g]))))):
                            i += 1
                            start = sum(tg[:i])
                            end = start + tg[i]
                            unit_predict[start:end, g] = stage[i + 1]
                            adjusted = True
                    if adjusted:
                        tg[i] += tg.pop(i + 1)
                        stage.pop(i)
                        tg, stage = compress_stages(tg, stage)
                    else:
                        i += 1
                # 极端情况判定
                if len(stage) == 2:
                    if tg[0] < min_start[g]:
                        unit_predict[:, g] = 1
                    # 额外检查逻辑
                for t in range(1, T):
                    if unit_predict[t, g] != unit_predict[t - 1, g]:  # 状态变化
                        # 检查先开后关或先关后开且违反约束的情况
                        if (unit_predict[t - 1, g] == 1 and unit_predict[t, g] == 0) and (tg[i - 1] < min_start[g]):
                            unit_predict[:, g] = 1  # 设置为全开
                            break
                        elif (unit_predict[t - 1, g] == 0 and unit_predict[t, g] == 1) and (tg[i - 1] < min_end[g]):
                            unit_predict[:, g] = 1  # 设置为全开
                            break
    return unit_predict

