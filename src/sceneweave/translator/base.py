"""行动翻译器的可注入接口。

翻译器把通过检查的声明转换为时间安排：新行动获得开始、过程节点与
结束安排并保留提交组；明确修改的行动获得只调整未来节点的安排。
本模块只定义契约，模型驱动的翻译实现不在当前交付内。
"""

from typing import Protocol

from sceneweave.domain.translation import TranslationRequest, TranslationResult


class Translator(Protocol):
    """翻译器契约；实现负责结合客观上下文判断具体时间安排。

    translate 在全部角色通过检查屏障之后、统一登记之前被调用；
    其间剧情时间冻结。返回结果必须关联同一响应；数据类型错误、
    缺字段或模型异常属于运行错误，不会被当作语义打回。
    """

    async def translate(self, request: TranslationRequest) -> TranslationResult: ...
