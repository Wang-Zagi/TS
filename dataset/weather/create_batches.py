import pandas as pd
import numpy as np
from typing import Generator, Dict, Tuple

filename_t_30min = 'T_30min.csv'
filename_10min = 'GroupA_10min.csv'
filename_120min = 'GroupB_120min.csv'

# 定义文件路径，假设文件存在于当前工作目录下
T_DEGC_FILE = 'T_degC_30min.csv'
GROUP_A_FILE = 'GroupA_10min.csv'
GROUP_B_FILE = 'GroupB_120min.csv'


def create_mixed_batches(
        history_len: int = 192,  # T (30min) 历史窗口长度
        future_len: int = 96,  # T (30min) 未来目标长度
        step_size: int = 96
) -> Generator[Tuple[Dict[str, np.ndarray], np.ndarray], None, None]:
    """
    读取多频率数据，构造 Seq2Seq 格式的训练批次 (X: 192, Y: 96)。

    - 输入是三个不同频率的 DataFrame (内部读取)。
    - 输出的 X (历史输入) 是一个包含三个不同形状 NumPy 数组的字典。

    生成器返回:
        (X_input_dict, Y_target_array)
        X_input_dict: 包含 T(192), A(574), B(48) 历史数据的字典。
        Y_target_array: 包含 T(96) 未来目标值的 NumPy 数组。
    """
    try:
        # 1. 读取数据并确保 DatetimeIndex
        df_t = pd.read_csv(T_DEGC_FILE, index_col=0, parse_dates=True)[['T (degC)']]
        df_a = pd.read_csv(GROUP_A_FILE, index_col=0, parse_dates=True)
        df_b = pd.read_csv(GROUP_B_FILE, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件。请确保以下文件存在: {T_DEGC_FILE}, {GROUP_A_FILE}, {GROUP_B_FILE}")
        return

    # --- 2. 批次构造逻辑 ---

    STEP_SIZE = step_size
    TOTAL_WINDOW = history_len + future_len
    n_t = len(df_t)

    # 修正后的期望长度 (Group A: 574, Group B: 48)
    expected_t_history_len = history_len
    expected_t_future_len = future_len
    expected_a_len = (history_len - 1) * 3 + 1
    expected_b_len = history_len // 4

    # 迭代 T (degC) 数据
    for i in range(0, n_t - TOTAL_WINDOW + 1, STEP_SIZE):

        # --- 定义时间窗口 ---
        history_start_time = df_t.index[i]
        history_end_time = df_t.index[i + history_len - 1]
        future_start_index = i + history_len

        # --- 构造历史输入 (X) ---
        batch_t_history = df_t.iloc[i: i + history_len].values
        batch_a_history = df_a.loc[history_start_time:history_end_time].values
        batch_b_history = df_b.loc[history_start_time:history_end_time].values

        # --- 构造未来目标 (Y) ---
        batch_t_future = df_t.iloc[future_start_index: future_start_index + future_len].values

        # --- 验证并生成批次 ---
        if (batch_t_history.shape[0] == expected_t_history_len and
                batch_a_history.shape[0] == expected_a_len and
                batch_b_history.shape[0] == expected_b_len and
                batch_t_future.shape[0] == expected_t_future_len):

            yield (
                {
                    'T_30min_hist': batch_t_history,
                    'A_10min_hist': batch_a_history,
                    'B_120min_hist': batch_b_history
                },
                batch_t_future
            )
        else:
            # 只有在数据对齐出现意外偏差时才打印警告
            print(
                f"Warning: Skipping misaligned batch starting at {history_start_time}. Shapes found: T_hist={batch_t_history.shape[0]}, A_hist={batch_a_history.shape[0]}, B_hist={batch_b_history.shape[0]}, T_fut={batch_t_future.shape[0]}")
            continue


# --- 使用示例 (与 create_aligned_batches 相同的使用方式) ---
if __name__ == '__main__':

    mixed_batch_generator = create_mixed_batches()

    try:
        # 获取第一个批次
        X_mixed_hist, Y_mixed_fut = next(mixed_batch_generator)

        print("\n--- 首个混合频率批次结构 ---")
        print(f"X (历史输入) 类型: {type(X_mixed_hist)}")
        print(f"X (T 30min) 形状: {X_mixed_hist['T_30min_hist'].shape}")
        print(f"X (A 10min) 形状: {X_mixed_hist['A_10min_hist'].shape}")
        print(f"X (B 120min) 形状: {X_mixed_hist['B_120min_hist'].shape}")
        print(f"Y (未来目标) 形状: {Y_mixed_fut.shape}")

    except StopIteration:
        print("\n生成器已耗尽，或文件不存在导致无法开始。")
    except Exception as e:
        print(f"\n运行中发生错误: {e}")




def create_aligned_batches(
        history_len: int = 192,
        future_len: int = 96,
        step_size: int = 96
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    读取多频率数据，统一对齐到 30 分钟频率，并构造 Seq2Seq 格式批次。

    步骤:
    1. 读取 T (30min), A (10min), B (120min) 数据。
    2. 对 Group A (10min) 下采样到 30min (丢弃多余数据)。
    3. 对 Group B (120min) 上采样到 30min (线性插值)。
    4. 合并所有数据。
    5. 使用滑动窗口构造历史 (X) 和未来目标 (Y) 批次。

    生成器返回:
        (X_history_batch, Y_future_target)
        X_history_batch.shape: (history_len, total_features)
        Y_future_target.shape: (future_len, 1) # 仅 T (degC)
    """
    try:
        # 1. 读取数据并确保 DatetimeIndex
        df_t = pd.read_csv(filename_t_30min, index_col=0, parse_dates=True)[['T (degC)']]
        df_a = pd.read_csv(filename_10min, index_col=0, parse_dates=True)
        df_b = pd.read_csv(filename_120min, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件。请确保以下文件存在: {filename_t_30min}, {filename_10min}, {filename_120min}")
        return

    # --- 2 & 3. 频率对齐 ---

    # Group A (10min -> 30min) 下采样: 丢弃多余数据 (first())
    # 确保对齐起始点 (origin='start')
    df_a_30min = df_a.resample('30min', origin='start').first()

    # Group B (120min -> 30min) 上采样: 线性插值
    # 先 resample 生成 30min 索引，再插值填充 NaN
    df_b_30min = df_b.resample('30min', origin='start').interpolate(method='linear')

    # --- 4. 合并数据 ---

    # 按照 T, A, B 的顺序合并所有数据，并删除插值和对齐可能产生的 NaN 行
    df_unified = pd.concat([df_t, df_a_30min, df_b_30min], axis=1).dropna()

    print(f"数据统一到 30min 频率，总形状: {df_unified.shape}")

    # 获取 T (degC) 所在的列索引 (通常是 0)
    t_col_index = df_unified.columns.get_loc('T (degC)')

    # --- 5. 构造批次 ---

    total_window = history_len + future_len
    n_total = len(df_unified)

    if n_total < total_window:
        print(f"数据总长度 ({n_total}) 小于所需窗口长度 ({total_window})。无法生成批次。")
        return

    for i in range(0, n_total - total_window + 1, step_size):

        # 历史输入 (X): 所有变量
        history_batch = df_unified.iloc[i: i + history_len].values

        # 未来目标 (Y): 仅 T (degC)
        future_start = i + history_len
        future_end = i + history_len + future_len

        # 切片 Y 目标 (仅 T 变量)
        target_batch = df_unified.iloc[future_start: future_end, t_col_index:t_col_index + 1].values

        # 检查最终批次形状是否正确
        if history_batch.shape[0] == history_len and target_batch.shape[0] == future_len:
            yield (history_batch, target_batch)
        else:
            # 这通常不应该发生，除非数据结尾不完整
            print(f"警告: 批次 {i} 附近出现不完整窗口，跳过。")