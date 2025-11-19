import pandas as pd
import numpy as np
import os

# --- 数据加载和预处理 (DatetimeIndex 转换) ---
df = pd.read_csv('weather.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
# ----------------------------------------------

# 变量分组定义
VAR_T_DEGC = ['T (degC)']
VAR_GROUP_10MIN = [
    'p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
    'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)'
]
VAR_GROUP_120MIN = [
    'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)',
    'raining (s)', 'SWDR (W/m�)', 'PAR (�mol/m�/s)',
    'max. PAR (�mol/m�/s)', 'Tlog (degC)', 'OT'
]

# 检查分组是否正确（非 T (degC) 的列总数应为 20）
if len(VAR_GROUP_10MIN) + len(VAR_GROUP_120MIN) != len(df.columns) - 1:
    print("Warning: Column count mismatch in defined groups!")

# --- 1. 处理 T (degC) 列 (30min 频率) ---
print("1. Processing T (degC) to 30min frequency...")
df_t_30min = df[VAR_T_DEGC].resample('30min', origin='start').first().dropna(how='all')
print(f"   T (degC) Final Shape: {df_t_30min.shape}")

# 保存为 CSV
filename_t_30min = 'T_30min.csv'
df_t_30min.to_csv(filename_t_30min)


# --- 2. 处理 Group A 变量 (保持 10min 频率) ---
print("\n2. Processing Group A to original 10min frequency...")
df_10min = df[VAR_GROUP_10MIN].copy()
print(f"   Group A Final Shape: {df_10min.shape}")

# 保存为 CSV
filename_10min = 'GroupA_10min.csv'
df_10min.to_csv(filename_10min)


# --- 3. 处理 Group B 变量 (120min 频率) ---
print("\n3. Processing Group B to 120min frequency...")
# 使用 first() 取 2 小时窗口内的第一个值作为代表
df_120min = df[VAR_GROUP_120MIN].resample('120min', origin='start').first().dropna(how='all')
print(f"   Group B Final Shape: {df_120min.shape}")

# 保存为 CSV
filename_120min = 'GroupB_120min.csv'
df_120min.to_csv(filename_120min)

print("\n--- Summary ---")
print(f"文件 '{filename_t_30min}' 已保存 (频率: 30min)")
print(f"文件 '{filename_10min}' 已保存 (频率: 10min)")
print(f"文件 '{filename_120min}' 已保存 (频率: 120min)")
print("数据处理和文件保存完毕。")

# 打印最终形状用于核对
print("\nShape Check:")
print(f"10min (Group A): {df_10min.shape}")
print(f"30min (T (degC)): {df_t_30min.shape}")
print(f"120min (Group B): {df_120min.shape}")

