# pairwise-aniso-cone-footprint

## 领域

游戏、图形与多媒体

## 主题

各向异性过滤锥体足迹

## 说明

接缝连续、椭圆轴长与 mip 链补齐。

初始状态下 `graphics/aniso_cone_footprint.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
