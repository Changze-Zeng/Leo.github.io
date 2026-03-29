# Leo.github.io
# -*- coding: utf-8 -*-
"""
生理年龄预测模型 - 预测与评估模块（含健康指标报警及拟合线绘制）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ========================== 1. 模型与标准化器加载 ==========================
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("模型加载成功。")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请先运行训练脚本生成 model.pkl 和 scaler.pkl 文件。")
    exit(1)

# ========================== 2. 特征定义 ==========================
FEATURES = [
    'RIAGENDR',   # 性别 (1=男, 2=女)
    'PAQ605',     # 体力活动 (1=是, 2=否)
    'BMXBMI',     # 身体质量指数 (kg/m²)
    'LBXGLU',     # 血糖 (mg/dL)
    'DIQ010',     # 糖尿病诊断 (1=是, 2=否, 3=边界)
    'LBXGLT',     # 血糖耐量测试 (mg/dL)
    'LBXIN'       # 胰岛素 (μU/mL)
]

# 年龄组划分
AGE_BINS = [0, 30, 50, 100]
AGE_LABELS = ['young', 'middle', 'senior']


# ========================== 3. 健康阈值配置 ==========================
class HealthThresholds:
    """健康指标阈值配置（基于专业医学文献）"""
    
    BMI_UNDERWEIGHT = 18.5
    BMI_NORMAL_MAX = 24.9
    BMI_OVERWEIGHT = 25.0
    BMI_OBESITY = 30.0
    
    GLUCOSE_LOW = 70
    GLUCOSE_NORMAL_MAX = 100
    GLUCOSE_PREDIABETES = 126
    
    GLT_NORMAL_MAX = 140
    GLT_IMPAIRED = 200
    
    INSULIN_NORMAL_MAX = 25


# ========================== 4. 健康报警函数 ==========================
def check_health_alerts(features_dict):
    """
    检查输入特征是否存在健康异常，返回报警信息列表。
    """
    alerts = []
    
    # 1. 体力活动检查
    if features_dict.get('PAQ605') == 2:
        alerts.append("⚠️ 体力活动不足：过去一个月无体力活动，建议每周进行150分钟中等强度运动。")
    
    # 2. BMI检查
    bmi = features_dict.get('BMXBMI')
    if bmi is not None:
        if bmi < HealthThresholds.BMI_UNDERWEIGHT:
            alerts.append(f"⚠️ 体重过轻：BMI = {bmi:.1f}，低于正常范围下限18.5。")
        elif bmi >= HealthThresholds.BMI_OBESITY:
            alerts.append(f"⚠️ 肥胖：BMI = {bmi:.1f}，≥30，心血管疾病风险升高。")
        elif bmi >= HealthThresholds.BMI_OVERWEIGHT:
            alerts.append(f"⚠️ 超重：BMI = {bmi:.1f}，建议控制饮食、增加运动。")
    
    # 3. 空腹血糖检查
    glucose = features_dict.get('LBXGLU')
    if glucose is not None:
        if glucose < HealthThresholds.GLUCOSE_LOW:
            alerts.append(f"⚠️ 低血糖：血糖 = {glucose:.1f} mg/dL，低于正常范围下限70。")
        elif glucose >= HealthThresholds.GLUCOSE_PREDIABETES:
            alerts.append(f"⚠️ 疑似糖尿病：血糖 = {glucose:.1f} mg/dL ≥ 126，建议立即就医。")
        elif glucose >= HealthThresholds.GLUCOSE_NORMAL_MAX:
            alerts.append(f"⚠️ 血糖偏高：血糖 = {glucose:.1f} mg/dL，处于糖尿病前期范围（100-125），建议控制饮食。")
    
    # 4. 糖尿病诊断状态
    diabetes = features_dict.get('DIQ010')
    if diabetes == 1:
        alerts.append("⚠️ 已有糖尿病诊断，请遵医嘱进行治疗和监测。")
    elif diabetes == 3:
        alerts.append("⚠️ 糖耐量异常（糖尿病边界），建议控制饮食、定期复查血糖。")
    
    # 5. 糖耐量测试检查
    glt = features_dict.get('LBXGLT')
    if glt is not None:
        if glt >= HealthThresholds.GLT_IMPAIRED:
            alerts.append(f"⚠️ 疑似糖尿病：餐后血糖 = {glt:.1f} mg/dL ≥ 200，建议立即就医。")
        elif glt >= HealthThresholds.GLT_NORMAL_MAX:
            alerts.append(f"⚠️ 糖耐量异常：餐后血糖 = {glt:.1f} mg/dL，处于140-199范围，建议控制饮食。")
    
    # 6. 胰岛素检查
    insulin = features_dict.get('LBXIN')
    if insulin is not None and insulin > HealthThresholds.INSULIN_NORMAL_MAX:
        alerts.append(f"⚠️ 胰岛素偏高：胰岛素 = {insulin:.1f} μU/mL > 25，提示可能存在胰岛素抵抗，建议进一步检查。")
    
    # 7. 数据有效性验证
    gender = features_dict.get('RIAGENDR')
    if gender is not None and gender not in [1, 2]:
        alerts.append("⚠️ 性别数据无效，请输入1（男）或2（女）。")
    
    return alerts


# ========================== 5. 预测函数（含报警） ==========================
def predict_physiological_age(features_dict):
    """
    根据输入的特征值预测生理年龄和年龄段，并检测健康异常。
    
    参数:
        features_dict: dict，键为特征名，值为对应的数值
    
    返回:
        dict: 包含预测年龄、年龄组和报警信息的字典
    """
    # 检查输入是否包含所有特征
    for f in FEATURES:
        if f not in features_dict:
            raise ValueError(f"缺少必要特征: {f}")
    
    # 健康报警检查
    alerts = check_health_alerts(features_dict)
    
    # 将输入转换为DataFrame，并确保列顺序与训练时一致（避免警告）
    input_df = pd.DataFrame([features_dict])[FEATURES]
    
    # 数据标准化（使用训练时保存的scaler，传入DataFrame保留特征名，消除警告）
    input_scaled = scaler.transform(input_df)   # 返回 numpy 数组
    
    # 预测年龄
    predicted_age = model.predict(input_scaled)[0]
    
    # 确定年龄段
    if predicted_age < AGE_BINS[1]:
        age_group = AGE_LABELS[0]
    elif predicted_age < AGE_BINS[2]:
        age_group = AGE_LABELS[1]
    else:
        age_group = AGE_LABELS[2]
    
    return {
        'predicted_age': round(predicted_age, 1),
        'age_group': age_group,
        'alerts': alerts
    }


# ========================== 6. 评估与绘图功能 ==========================
def evaluate_and_plot(test_data_path, target_col='RIDAGEYR', sample_frac=1.0):
    """
    加载测试数据集，进行预测，绘制真实年龄 vs 预测年龄的拟合线。
    
    参数:
        test_data_path: 测试数据文件路径（CSV格式）
        target_col: 真实年龄列名，默认为 'RIDAGEYR'
        sample_frac: 采样比例（数据量大时可降低绘图密度），默认1.0
    """
    # 加载测试数据
    df_test = pd.read_csv(test_data_path)
    print(f"测试数据形状: {df_test.shape}")
    
    # 检查必需列
    missing_features = set(FEATURES) - set(df_test.columns)
    if missing_features:
        raise ValueError(f"测试数据缺少特征列: {missing_features}")
    if target_col not in df_test.columns:
        raise ValueError(f"测试数据缺少目标列: {target_col}")
    
    # 提取特征和目标
    X_test = df_test[FEATURES].copy()
    y_true = df_test[target_col].copy()
    
    # 预处理（缺失值用中位数填充，与训练时保持一致）
    # 注意：这里未保存训练时的imputer，因此重新拟合（但预测时也需一致，此处仅用于评估）
    num_imputer = SimpleImputer(strategy='median')
    X_test_imputed = num_imputer.fit_transform(X_test)
    
    # 标准化
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n测试集评估:")
    print(f"  均方误差 (MSE): {mse:.2f}")
    print(f"  均方根误差 (RMSE): {rmse:.2f} 岁")
    print(f"  平均绝对误差 (MAE): {mae:.2f} 岁")
    print(f"  R² 分数: {r2:.4f}")
    
    # 可选：随机采样以减少绘图点（如果数据量很大）
    if sample_frac < 1.0:
        n_samples = len(y_true)
        idx = np.random.choice(n_samples, int(n_samples * sample_frac), replace=False)
        y_true_sample = y_true.iloc[idx]
        y_pred_sample = y_pred[idx]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    # 绘制散点图及拟合线（y=x）
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=20, edgecolors='none')
    # 对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x (理想拟合)')
    plt.xlabel('真实年龄 (岁)')
    plt.ylabel('预测年龄 (岁)')
    plt.title(f'真实年龄 vs 预测年龄 (RMSE={rmse:.2f}岁, R²={r2:.3f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========================== 7. 辅助函数：特征说明 ==========================
def print_feature_info():
    """
    打印所有特征的说明信息，包括医学参考范围。
    """
    print("\n" + "="*60)
    print("输入特征说明及医学参考范围")
    print("="*60)
    
    print("\n【RIAGENDR - 性别】")
    print("  取值: 1=男性, 2=女性")
    print("  用途: 统计分组")
    
    print("\n【PAQ605 - 体力活动】")
    print("  取值: 1=是（过去一个月有体力活动）, 2=否")
    print("  正常: 1（有体力活动）")
    print("  报警: 值为2时提示缺乏运动")
    
    print("\n【BMXBMI - 身体质量指数】")
    print("  正常范围: 18.5 ~ 24.9 kg/m²")
    print("  超重: 25.0 ~ 29.9 kg/m²")
    print("  肥胖: ≥ 30.0 kg/m²")
    
    print("\n【LBXGLU - 空腹血糖】")
    print("  正常范围: 70 ~ 100 mg/dL")
    print("  糖尿病前期: 100 ~ 125 mg/dL")
    print("  糖尿病: ≥ 126 mg/dL")
    
    print("\n【DIQ010 - 糖尿病诊断】")
    print("  取值: 1=是, 2=否, 3=边界")
    
    print("\n【LBXGLT - 血糖耐量测试/餐后血糖】")
    print("  正常范围: < 140 mg/dL")
    print("  糖耐量异常: 140 ~ 199 mg/dL")
    print("  糖尿病: ≥ 200 mg/dL")
    
    print("\n【LBXIN - 胰岛素】")
    print("  正常范围: 2 ~ 25 μU/mL（空腹）")
    print("  胰岛素抵抗: > 25 μU/mL")
    
    print("\n" + "="*60)


# ========================== 8. 主程序 ==========================
if __name__ == "__main__":
    # ========== 模式选择 ==========
    # 将此变量改为 True 即可执行评估绘图模式，改为 False 执行交互式预测模式
    RUN_EVAL = True   # 默认 False，即交互式预测模式
    
    if RUN_EVAL:
        # 评估绘图模式 - 在这里设置测试数据文件路径等参数
        test_data_path = 'data.csv'          # 测试数据文件路径（请修改为实际路径）
        target_column = 'RIDAGEYR'           # 目标列名
        sample_frac = 1.0                    # 采样比例（1.0表示使用全部数据）
        evaluate_and_plot(test_data_path, target_col=target_column, sample_frac=sample_frac)
    else:
        # 交互式预测模式（原代码）
        print_feature_info()
        print("\n请输入以下特征值进行预测（按回车使用默认示例值）：")
        
        default_values = {
            'RIAGENDR': 1,
            'PAQ605': 2,
            'BMXBMI': 28.5,
            'LBXGLU': 105.0,
            'DIQ010': 2,
            'LBXGLT': 145.0,
            'LBXIN': 18.5
        }
        
        sample = {}
        for feat in FEATURES:
            default = default_values[feat]
            try:
                if feat in ['RIAGENDR', 'PAQ605', 'DIQ010']:
                    val = input(f"{feat} (默认 {default}): ").strip()
                    sample[feat] = int(val) if val else default
                else:
                    val = input(f"{feat} (默认 {default}): ").strip()
                    sample[feat] = float(val) if val else default
            except ValueError:
                print(f"输入无效，使用默认值 {default}")
                sample[feat] = default
        
        # 预测
        result = predict_physiological_age(sample)
        
        # 输出结果
        print(f"\n{'='*50}")
        print("预测结果")
        print(f"{'='*50}")
        print(f"生理年龄: {result['predicted_age']} 岁")
        print(f"年龄段: {result['age_group']}")
        
        if result['alerts']:
            print(f"\n健康报警:")
            for alert in result['alerts']:
                print(f"  {alert}")
        else:
            print(f"\n健康指标正常，无异常报警。")
        print(f"{'='*50}")

