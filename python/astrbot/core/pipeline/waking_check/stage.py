from ..stage import Stage, register_stage
from ..context import PipelineContext
from typing import Union, AsyncGenerator
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageEventResult, MessageChain
from astrbot.core.message.components import At, Reply
from astrbot.core.star.star_handler import star_handlers_registry, EventType
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.filter.permission import PermissionTypeFilter

@register_stage
class WakingCheckStage(Stage):
    """检查是否需要唤醒。唤醒机器人有如下几点条件：

    1. 机器人被 @ 了
    2. 机器人的消息被提到了
    3. 以 wake_prefix 前缀开头，并且消息没有以 At 消息段开头
    4. 插件（Star）的 handler filter 通过
    5. 私聊情况下，位于 admins_id 列表中的管理员的消息（在白名单阶段中）
    """

    async def initialize(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        self.no_permission_reply = self.ctx.astrbot_config["platform_settings"].get(
            "no_permission_reply", True
        )

    async def process(
        self, event: AstrMessageEvent
    ) -> Union[None, AsyncGenerator[None, None]]:
        # 设置 sender 身份
        event.message_str = event.message_str.strip()
        for admin_id in self.ctx.astrbot_config["admins_id"]:
            if event.get_sender_id() == admin_id:
                event.role = "admin"
                break

        # 检查 wake
        wake_prefixes = self.ctx.astrbot_config["wake_prefix"]
        messages = event.get_messages()
        is_wake = False
        for wake_prefix in wake_prefixes:
            if event.message_str.startswith(wake_prefix):
                if (
                    not event.is_private_chat()
                    and isinstance(messages[0], At)
                    and str(messages[0].qq) != str(event.get_self_id())
                    and str(messages[0].qq) != "all"
                ):
                    # 如果是群聊，且第一个消息段是 At 消息，但不是 At 机器人或 At 全体成员，则不唤醒
                    break
                is_wake = True
                event.is_at_or_wake_command = True
                event.is_wake = True
                event.message_str = event.message_str[len(wake_prefix) :].strip()
                break
        if not is_wake:
            # 检查是否有 at 消息
            for message in messages:
                if isinstance(message, At) and (
                    str(message.qq) == str(event.get_self_id())
                    or str(message.qq) == "all"
                ):
                    is_wake = True
                    event.is_wake = True
                    wake_prefix = ""
                    event.is_at_or_wake_command = True
                    break
            # 检查是否是私聊
            if event.is_private_chat():
                is_wake = True
                event.is_wake = True
                event.is_at_or_wake_command = True
                wake_prefix = ""

        # 检查插件的 handler filter
        activated_handlers = []
        handlers_parsed_params = {}  # 注册了指令的 handler
        for handler in star_handlers_registry.get_handlers_by_event_type(EventType.AdapterMessageEvent):
            # filter 需要满足 AND 的逻辑关系
            passed = True
            child_command_handler_md = None
            
            permission_not_pass = False
            
            if len(handler.event_filters) == 0:
                # 不可能有这种情况, 也不允许有这种情况
                continue
            
            if 'sub_command' in handler.extras_configs:
                # 如果是子指令
                continue

            for filter in handler.event_filters:
                try:
                    if isinstance(filter, CommandGroupFilter):
                        """如果指令组过滤成功, 会返回叶子指令的 StarHandlerMetadata"""
                        ok, child_command_handler_md = filter.filter(
                            event, self.ctx.astrbot_config
                        )
                        if not ok:
                            passed = False
                        else:
                            handler = child_command_handler_md  # handler 覆盖
                            break
                    elif isinstance(filter, PermissionTypeFilter):
                        if not filter.filter(event, self.ctx.astrbot_config):
                            permission_not_pass = True
                    else:
                        if not filter.filter(event, self.ctx.astrbot_config):
                            passed = False
                            break
                except Exception as e:
                    # event.set_result(MessageEventResult().message(f"插件 {handler.handler_full_name} 报错：{e}"))
                    # yield
                    await event.send(
                        MessageEventResult().message(
                            f"插件 {handler.handler_full_name} 报错：{e}"
                        )
                    )
                    event.stop_event()
                    passed = False
                    break

            if passed:
                
                if permission_not_pass:
                    if self.no_permission_reply:
                        await event.send(MessageChain().message(f"ID {event.get_sender_id()} 权限不足。通过 /sid 获取 ID 并请管理员添加。"))
                    event.stop_event()
                    return
                
                is_wake = True
                event.is_wake = True

                activated_handlers.append(handler)
                if "parsed_params" in event.get_extra():
                    handlers_parsed_params[handler.handler_full_name] = event.get_extra(
                        "parsed_params"
                    )
            event.clear_extra()

        event.set_extra("activated_handlers", activated_handlers)
        event.set_extra("handlers_parsed_params", handlers_parsed_params)

        if not is_wake:
            event.stop_event()
