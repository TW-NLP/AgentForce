"""
工作流回调处理模块
管理状态回调和事件分发
"""

import asyncio
import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class EventType:
    """事件类型常量"""
    # 工作流步骤事件
    STEP = "step"
    
    # LLM 生成事件
    TOKEN = "token"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    
    # 进度事件
    PROGRESS = "progress"
    
    # 错误和警告
    ERROR = "error"
    WARNING = "warning"
    
    # 完成事件
    DONE = "done"


class StepEvent:
    """步骤事件数据结构"""
    
    @staticmethod
    def create(step: str, title: str, description: str) -> Dict[str, Any]:
        return {
            "step": step,
            "title": title,
            "description": description
        }


class TokenEvent:
    """Token 事件数据结构"""
    @staticmethod
    def create(content: str, full_message: str = "") -> Dict[str, Any]:
        return {
            "content": content,
            "full_message": full_message
        }


class StatusCallback:
    """状态回调管理器"""
    
    def __init__(self):
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """移除回调函数"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """触发所有回调 (核心分发逻辑)"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"回调执行错误: {e}", exc_info=True)

    # =================================================================
    # ✅ 新增：适配 Agent 生命周期的专用方法
    # 这些方法将 Agent 的行为转换为前端能理解的 "StepEvent"
    # =================================================================

    async def on_agent_start(self, data: Dict[str, Any]):
        """当 Agent 开始处理任务时调用"""
        await self.emit(EventType.STEP, StepEvent.create(
            step="init",
            title="开始思考",
            description=f"收到任务: {data.get('input', '')[:50]}..."
        ))

    async def on_tool_start(self, data: Dict[str, Any]):
        """当 Agent 决定调用工具时调用"""
        tool_name = data.get("name", "Unknown Tool")
        args = data.get("args", {})
        
        await self.emit(EventType.STEP, StepEvent.create(
            step="tool_start",
            title=f"调用工具: {tool_name}",
            description=f"参数: {str(args)[:100]}..."
        ))

    async def on_tool_end(self, data: Dict[str, Any]):
        """当工具执行完毕返回结果时调用"""
        output = data.get("output", "")
        
        await self.emit(EventType.STEP, StepEvent.create(
            step="tool_end",
            title="工具执行完成",
            description=f"结果预览: {str(output)[:100]}..."
        ))

    async def on_agent_finish(self, data: Dict[str, Any]):
        """当 Agent 完成所有步骤准备回答时调用"""
        await self.emit(EventType.STEP, StepEvent.create(
            step="finish",
            title="流程结束",
            description="任务已完成，正在生成最终回复"
        ))
        
    async def on_llm_new_token(self, token: str):
        """(可选) 如果需要流式输出 Token"""
        await self.emit(EventType.TOKEN, TokenEvent.create(
            content=token
        ))

    def clear(self):
        """清空所有回调"""
        self.callbacks.clear()