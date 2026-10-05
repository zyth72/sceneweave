"""角色内部组件的基础协议：模型采样接口。

组件通过可注入的采样器获得模型文本；采样器实现负责实际调用（脚本
假件用于测试，Concordia 语言模型桥接见 integrations 层）。构造或
导入不启动模型、不读取配置；超时/错误/取消如实传播，组件不得吞掉
异常或悄悄改用假回答。
"""

from typing import Protocol


class TextSampler(Protocol):
    """按组件名采一段模型文本；实现必须可异步等待且不阻塞事件循环。"""

    async def sample(self, prompt: str, *, component: str) -> str: ...
