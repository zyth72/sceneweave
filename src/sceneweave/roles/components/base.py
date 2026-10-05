"""角色内部组件的基础协议：模型采样接口。

协议已提取到 sceneweave.domain.modeling（角色与翻译器共用）；本模块
再导出以保持既有导入路径兼容。
"""

from sceneweave.domain.modeling import TextSampler

__all__ = ["TextSampler"]
