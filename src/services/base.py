import logging
from abc import ABC, abstractmethod
from typing import Any
from config import settings  # 导入全局 settings 单例

logger = logging.getLogger(__name__)

class BaseConfigurableService(ABC):
    """
    可配置服务的基类
    自动处理配置变更检测和热重载
    """
    
    def __init__(self):
        self.settings = settings  # 持有全局配置的引用
        self._last_config_hash = None
        self._instance = None     

    @property
    def config_hash(self) -> str:
        """获取当前全局配置的指纹"""
        return self.settings.config_hash

    @abstractmethod
    def build_instance(self) -> Any:
        """
        【子类必须实现】
        定义如何利用当前的 self.settings 构建业务实例
        """
        pass

    def get_instance(self) -> Any:
        """
        获取业务实例 (带热重载逻辑)
        外部调用时，不要直接用 self.agent，而是调用这个方法
        """
        current_hash = self.config_hash
        
        # 如果是第一次初始化，或者配置发生了变化
        if self._instance is None or self._last_config_hash != current_hash:
            logger.info(f"🔄 配置已变更 (Old: {self._last_config_hash}, New: {current_hash})，正在重建服务...")
            
            try:
                # 调用子类的构建逻辑
                self._instance = self.build_instance()
                # 更新指纹
                self._last_config_hash = current_hash
                logger.info("✅ 服务重建成功")
            except Exception as e:
                logger.error(f"❌ 服务重建失败: {e}")
                # 如果重建失败，且旧实例存在，则降级使用旧实例（防止服务彻底挂掉）
                if self._instance:
                    logger.warning("⚠️ 降级使用旧的实例")
                    return self._instance
                raise e

        return self._instance