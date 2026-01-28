# 老年健康服务需求预测（ADL 0–6）——回归与有序分类基线

**目标**：预测个体 ADL 等级（0–6）。数据包含 0/1 二值特征（是否患癌、是否吸烟等）与连续特征（如退休金）。

## 任务定义
- **回归**（推荐作为主线）：ADL=0..6 等距，回归→四舍五入输出等级。
- **有序分类**：将 0..6 的顺序通过累计链接（cumulative link）建模（K-1 个 y>k 的二分类器），比普通多分类更符合业务语义。

## 指标
- 回归：MAE、RMSE、R²；并报告 Acc_rounded（取整后准确率）、Within1（|误差|≤1）
- 有序分类：Acc、MAE_on_classes、Within1、QWK（二次加权Kappa）

## 使用
### 1) 训练脚本
```bash
python train_regression_and_ordinal.py --data <你的CSV路径> --target adlab-c
```
- 若目标列不同，请通过 `--target` 指定；未指定将自动识别（列名包含“adl”且为 0..6）。
- 输出：`metrics_summary.csv`、`confusion_matrix.png`、`target_hist.png`。

### 2) 一键 Notebook
- 打开 `ADL_end2end.ipynb`，在第一格设置 `DATA_PATH` 与（可选）`TARGET_HINT`，按顺序运行。

## 流程要点
1. 缺失值：二值→众数；连续→中位数
2. 切分：对 0–6 分层（Stratified split）
3. 回归：ElasticNet（带标准化）与 HistGradientBoostingRegressor
4. 有序分类：训练 6 个 y>k 的二分类器（HGBClassifier），推理还原 P(y=c)
5. 评估：统一测试集对比，业务按 KPI（QWK/Within1 或 MAE/RMSE）选择方案

## 注意
- 仅使用 matplotlib（无 seaborn），图像一图一画布。
- 若类别极不均衡，建议在训练时加入样本权重或重采样作为增强实验。
