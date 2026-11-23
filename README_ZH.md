# When Does History Transfer?（CIFEr 核心版）

[English](README.md)

当时间序列发生机制切换时，借用相关序列的历史预测误差何时能降低误差，何时又该回退到本地基线？

本仓库提供 *When Does History Transfer?*（IEEE SSCI 2027 完整论文录用、NeurIPS 2026 FMTS Workshop 录用）的 CIFEr 核心实现。系统把历史误差按上下文相似度检索出来，显式区分误差来源，再通过残差源岭回归层做信任校准与回退门控，避免在机制不匹配时盲目迁移错误。

- `src/core.py`：上下文检索、误差来源拆解与预测评估工具。
- `src/residual_source_ridge.py`：带回退机制与信任校准的条件残差源岭回归实现。
- `src/paths.py`：本地运行的轻量路径配置。

## 快速开始

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` 在合成跨机制序列上运行残差源岭回归适配器，对照无条件误差复用与信任校准迁移的表现。

## 范围说明

本仓库为合成序列上的公开 MVP，完整跨市场波动率与宏观评测套件按需提供。
