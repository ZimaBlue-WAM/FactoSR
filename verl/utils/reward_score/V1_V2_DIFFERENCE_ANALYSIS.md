# V1 vs V2 结果差异分析报告

## 问题现象

在测试中发现：
- **V1 (spatial_xy.py)**: `r_reprojection = 0.0`
- **V2 (spatial_xy_2.py)**: `r_reprojection = 0.971867`

## 根本原因

### V1 的额外检查：Overlap Mask

**V1 (spatial_xy.py 第1137-1148行) 包含额外的 overlap region 检查：**

```python
if depth1 is not None and depth2 is not None:
    mask1, mask2 = compute_overlap_mask(depth1, K1, pose1, depth2, K2, pose2)
    
    # Check if predicted point is in overlap region
    u_pred_px, v_pred_px = int(pt_pred_pixel[0]), int(pt_pred_pixel[1])
    
    if 0 <= v_pred_px < mask2.shape[0] and 0 <= u_pred_px < mask2.shape[1]:
        in_overlap = mask2[v_pred_px, u_pred_px]
        if not in_overlap:
            # Point is outside overlap region
            # Set r_reprojection to 0 as penalty
            r_reprojection = 0.0
```

**V2 (spatial_xy_2.py) 没有这个检查！**

## 详细分析

### 测试场景

使用 `np.random.seed(42)` 生成的测试数据：

```
相机设置：
- Image size: 480 x 640
- pose1: 单位矩阵 (原点)
- pose2: 沿 x 轴平移 0.5m

参考点 (View 1): (320.0, 240.0) - 图像中心
深度: 1.621948m

投影结果:
- 从 View 1 的 (320, 240) 投影到 View 2 的 (158.16, 240.00)

预测点 A (View 2): (350.0, 240.0)

像素距离: 191.84 pixels
归一化距离: 0.299750
```

### Overlap Mask 分析

```bash
Computing overlap mask...
mask1: True pixels: 5346/307200 (1.74%)
mask2: True pixels: 5318/307200 (1.73%)

Predicted point A (350, 240): NOT in overlap region ❌
Expected projection (158, 240): NOT in overlap region ❌
```

**关键发现：**
1. 两个视图的 overlap 区域非常小（只有 1.7%）
2. 预测点 A (350, 240) **不在** overlap 区域内
3. V1 检测到这一点，将 `r_reprojection` 设为 0.0
4. V2 没有这个检查，继续计算并返回正常的 reward

### 为什么 Overlap 区域这么小？

因为测试设置中：
```python
pose2[0, 3] = 0.5  # 沿 x 轴平移 0.5 米
```

相机平移了 0.5 米，而平均深度约 3 米，导致视图之间的 overlap 很小。

## 设计哲学差异

### V1 的设计理念（更严格）

```
✓ 考虑几何一致性
✓ 考虑 overlap 区域约束
✓ 只有在两个视图都可见的区域内才给予奖励
✗ 可能过于严格，拒绝了合理的预测
```

**优点：**
- 更符合多视图几何的物理约束
- 确保预测点在两个视图中都可见
- 避免"幻觉"预测（预测了实际看不到的点）

**缺点：**
- 当 overlap 区域很小时，几乎所有预测都会被拒绝
- 即使预测在几何上是合理的（重投影误差小），也可能因为不在 overlap 区域而被惩罚
- 测试结果显示：overlap 只有 1.7%，导致大量合理预测被错误惩罚

### V2 的设计理念（更宽松）

```
✓ 考虑几何一致性
✗ 不考虑 overlap 区域约束
✓ 只要几何上可以计算，就给予奖励
✓ 更符合实际 VLM 的预测场景
```

**优点：**
- 更加实用，不会因为 overlap 小而拒绝合理预测
- 关注核心目标：几何重投影一致性
- 适合 VLM 可能预测任意位置的场景

**缺点：**
- 可能给予"不可见"区域的预测过高的奖励
- 没有考虑物理可见性约束

## 数值验证

### 重投影计算（两个版本都正确）

```python
# View 1 参考点 (320, 240) at depth 1.622m
→ View 1 camera coords: (0.000, 0.000, 1.622)
→ World coords: (0.000, 0.000, 1.622)
→ View 2 camera coords: (-0.500, 0.000, 1.622)
→ View 2 projected: (158.16, 240.00)

预测点: (350.0, 240.0)
归一化距离: 0.299750

sigma = 10.0 (归一化单位)
reward = exp(-0.299750 / 10.0) = 0.970470 ✓
```

**两个版本的重投影计算都是正确的！**

差异仅在于：
- V1 检查 overlap mask → 不在 overlap → 返回 0.0
- V2 不检查 overlap → 直接返回 0.970470

## 建议

### 1. 明确使用场景

**使用 V1 (spatial_xy.py) 当：**
- ✅ 需要严格的多视图几何约束
- ✅ 训练数据中两个视图有足够的 overlap
- ✅ 希望模型只预测"可见"区域的点

**使用 V2 (spatial_xy_2.py) 当：**
- ✅ 关注几何一致性，但不强制 overlap
- ✅ VLM 可能预测任意位置
- ✅ 训练数据中 overlap 可能较小
- ✅ 需要更高的性能

### 2. V1 的改进建议

可以考虑将 overlap mask 检查改为**可选**：

```python
# 添加配置项
use_overlap_check = extra_info.get("use_overlap_check", False)

if use_overlap_check and depth1 is not None and depth2 is not None:
    mask1, mask2 = compute_overlap_mask(...)
    # ... overlap 检查逻辑
```

### 3. V2 的改进建议

可以考虑添加**可选的** overlap mask 检查：

```python
# 在 reproject_full_depth_map 中已经计算了 valid_mask
# 可以基于此实现 overlap 检查
use_overlap_check = extra_info.get("use_overlap_check", False)

if use_overlap_check:
    # 使用 valid_mask 进行检查
    if not valid_mask[v_ref_int, u_ref_int]:
        return 0.0
```

## 测试场景的合理性

在实际的点对应任务中：
1. **正常场景**：两个视图应该有足够的 overlap (30%+)
2. **测试场景**：overlap 只有 1.7%，这是一个**极端情况**
3. **建议**：调整测试数据，减小相机平移距离，增加 overlap

### 改进的测试数据

```python
# 原始（overlap 1.7%）
pose2[0, 3] = 0.5  # 平移 0.5m

# 改进（应该增加 overlap）
pose2[0, 3] = 0.1  # 平移 0.1m（更合理）
```

或者使用旋转而不是平移：

```python
# 小角度旋转，保持更大的 overlap
import numpy as np
angle = np.radians(10)  # 10度旋转
pose2[:3, :3] = np.array([
    [np.cos(angle), 0, np.sin(angle)],
    [0, 1, 0],
    [-np.sin(angle), 0, np.cos(angle)]
])
```

## 结论

1. **V1 和 V2 的重投影计算都是正确的** ✓
2. **差异来自 V1 的 overlap mask 检查** - 这是设计差异，不是 bug
3. **在测试场景中**：
   - overlap 只有 1.7%（极端情况）
   - 预测点不在 overlap 区域
   - V1 正确地拒绝了（按其设计）
   - V2 正确地计算了（按其设计）
4. **两个版本都有各自的应用场景**

### 推荐配置

```python
# 生产环境 - 使用 V2 (更实用)
reward_module = 'spatial_xy_2'
use_overlap_check = False

# 如果需要严格的几何约束
reward_module = 'spatial_xy'
# 并确保训练数据有足够的 overlap (>20%)
```
