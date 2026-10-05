"""角色内部的 System1/2/3 组件实现；编排由角色负责。

组件通过可注入的采样器获得模型文本，阶段留痕写入与引擎相同的追踪
链；这里不依赖 Concordia（桥接见 integrations/concordia/sampling.py）。
"""
