# MuJoCo Panda抓取问题 - 解决方案总结

## 问题诊断

### 根本原因
1. **Panda手的45°z旋转**导致手指闭合方向计算复杂
2. **IK定位误差大**：手指中点无法精确对准方块中心（误差100mm+）
3. **手指几何形状**：手指pad位置和方块不对齐

### 具体表现
- 手指中点偏移方块中心100mm+
- 夹爪闭合时方块被推开
- 夹爪完全闭合后，手指-方块接触丢失

## 解决方案

### 方案1：修改场景XML（推荐）

修改 `/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml`，
将Panda手的安装方式改为0°旋转，这样手指闭合方向就沿着简单的轴。

**步骤**：
1. 打开 `panda.xml` 文件
2. 找到hand body的定义
3. 将手的quat从 `-0.3826834 0 0 0.9238795` 改为 `1 0 0 0`（无旋转）
4. 重新运行测试

### 方案2：使用MuJoCo的mocap功能

不使用IK，而是使用MuJoCO的mocap body来直接控制hand位置：

```python
# 创建mocap body
model.body('hand').mocapid = 0  # 设置mocap ID

# 在仿真中，直接设置mocap位置
data.mocap_pos[0] = target_pos
```

这样可以绕过IK，直接控制hand位置。

### 方案3：使用预计算的好位置 + 手动调整

1. 运行MuJoCo可视化仿真
2. 手动移动机械臂到方块上方
3. 记录关节角度
4. 在代码中使用这些关节角度

**示例代码**：参见 `test_grasp_preset.py`

### 方案4：调整抓取策略

从方块正上方抓取（垂直向下），而不是从侧面抓取：

1. 将方块放在桌子边缘
2. 从侧面接近方块（y方向）
3. 确保手指闭合方向对准方块

## 快速修复步骤

### 步骤1：验证手指几何形状

运行诊断脚本，检查手指pad位置：
```bash
cd /home/w/vla_workspace
python diagnose_pad_position.py
```

### 步骤2：检查碰撞属性

确保手指pad的contype和conaffinity设置正确：
- `lf_pad1`到`lf_pad5`：contype=2, conaffinity=2
- `rf_pad1`到`rf_pad5`：contype=2, conaffinity=2
- `red_block_geom`：contype=2, conaffinity=2

从之前的输出看，这些设置是正确的。

### 步骤3：增加摩擦和夹持力

在场景XML中增加摩擦和夹持力：
```xml
<!-- 增加方块摩擦 -->
<geom name="red_block_geom" friction="2 1 0.01"/>

<!-- 增加手指pad摩擦 -->
<geom name="lf_pad1" friction="2 1 0.01"/>
<!-- 对所有pad重复 -->
```

### 步骤4：使用更慢的夹爪闭合

在代码中，使用更慢的夹爪闭合速度，并监控接触力。

## 推荐的工作流程

1. **先让手指接触到方块**（即使不抬起）
2. **监控接触力**，确保手指和方块之间有接触
3. **慢慢抬起**，监控方块是否跟随

## 代码修改清单

### 修改1：修复IK定位
使用校准的hand-to-finger-mid偏移量，而不是假设固定值。
参见 `test_contact_force.py` 中的Phase 2修改。

### 修改2：改进夹爪闭合策略
在闭合夹爪的同时缓慢抬起，避免方块被推开。
参见 `test_contact_force.py` 中的Phase 3修改。

### 修改3：简化抬起策略
先确认方块被夹住，再尝试抬起。
参见 `test_grasp_simple.py`。

## 测试建议

1. **先测试简单场景**：将方块放在更容易抓取的位置
2. **使用可视化调试**：运行MuJoCo viewer，观察手指和方块的接触
3. **逐步验证**：先验证手指能接触到方块，再验证能抬起

## 参考资料

- MuJoCo文档：https://mujoco.readthedocs.io/
- Panda手模型：检查 `panda.xml` 中的手配置
- 接触参数：调整 `condim`, `friction`, `solref`, `solimp`

## 下一步行动

1. 修改场景XML，简化手的旋转
2. 重新运行测试
3. 如果仍失败，使用mocap功能直接控制hand位置
4. 考虑使用其他抓取策略（如吸盘）

---
生成时间：2026-06-26
