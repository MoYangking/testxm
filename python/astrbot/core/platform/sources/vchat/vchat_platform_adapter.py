import sys
import time
import uuid
import asyncio
import os

from astrbot.api.platform import Platform, AstrBotMessage, MessageMember, MessageType, PlatformMetadata
from astrbot.api.event import MessageChain
from astrbot.api.message_components import *
from astrbot.api import logger
from astrbot.core.platform.astr_message_event import MessageSesion
from .vchat_message_event import VChatPlatformEvent
from ...register import register_platform_adapter

from vchat import Core
from vchat import model

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@register_platform_adapter("vchat", "基于 VChat 的 Wechat 适配器")
class VChatPlatformAdapter(Platform):

    def __init__(self, platform_config: dict, platform_settings: dict, event_queue: asyncio.Queue) -> None:
        super().__init__(event_queue)
        self.config = platform_config
        self.settingss = platform_settings
        self.test_mode = os.environ.get('TEST_MODE', 'off') == 'on'
        self.client_self_id = uuid.uuid4().hex[:8]
    
    @override
    async def send_by_session(self, session: MessageSesion, message_chain: MessageChain):
        from_username = session.session_id.split('$$')[0]
        await VChatPlatformEvent.send_with_client(self.client, message_chain, from_username)
        await super().send_by_session(session, message_chain)
    
    @override
    def meta(self) -> PlatformMetadata:
        return PlatformMetadata(
            "vchat",
            "基于 VChat 的 Wechat 适配器",
        )

    @override
    def run(self):
        self.client = Core()
        @self.client.msg_register(msg_types=model.ContentTypes.TEXT, 
                                  contact_type=model.ContactTypes.CHATROOM | model.ContactTypes.USER)
        async def _(msg: model.Message):
            if isinstance(msg.content, model.UselessContent):
                return
            if msg.create_time < self.start_time:
                logger.debug(f"忽略旧消息: {msg}")
                return
            logger.debug(f"收到消息: {msg.todict()}")
            abmsg = self.convert_message(msg)
            # await self.handle_msg(abmsg) # 不能直接调用，否则会阻塞
            asyncio.create_task(self.handle_msg(abmsg))
        
        # TODO: 对齐微信服务器时间
        self.start_time = int(time.time())
        return self._run()
    

    async def _run(self):
        await self.client.init()
        await self.client.auto_login(hot_reload=True, enable_cmd_qr=True)
        await self.client.run()
    
    def convert_message(self, msg: model.Message) -> AstrBotMessage:
        # credits: https://github.com/z2z63/astrbot_plugin_vchat/blob/master/main.py#L49
        assert isinstance(msg.content, model.TextContent)
        amsg = AstrBotMessage()
        amsg.message = [Plain(msg.content.content)]
        amsg.self_id = self.client_self_id
        if msg.content.is_at_me:
            amsg.message.insert(0, At(qq=amsg.self_id))
        
        sender = msg.chatroom_sender or msg.from_
        amsg.sender = MessageMember(sender.username, sender.nickname)
        
        if msg.content.is_at_me:
            amsg.message_str = msg.content.content.split("\u2005")[1].strip()
        else:
            amsg.message_str = msg.content.content
        amsg.message_id = msg.message_id
        if isinstance(msg.from_, model.User):
            amsg.type = MessageType.FRIEND_MESSAGE
        elif isinstance(msg.from_, model.Chatroom):
            amsg.type = MessageType.GROUP_MESSAGE
            amsg.group_id = msg.from_.username
        else:
            logger.error(f"不支持的 Wechat 消息类型: {msg.from_}")
            
        amsg.raw_message = msg
        
        if self.settingss['unique_session']:
            session_id = msg.from_.username + "$$" + msg.to.username
            if msg.chatroom_sender is not None:
                session_id += '$$' + msg.chatroom_sender.username
        else:
            session_id = msg.from_.username
                
        amsg.session_id = session_id
        return amsg
    
    async def handle_msg(self, message: AstrBotMessage):
        message_event = VChatPlatformEvent(
            message_str=message.message_str,
            message_obj=message,
            platform_meta=self.meta(),
            session_id=message.session_id,
            client=self.client
        )
        
        logger.info(f"处理消息: {message_event}")
        
        self.commit_event(message_event)