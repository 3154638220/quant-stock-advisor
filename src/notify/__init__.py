"""外部通知渠道集成：企业微信 Webhook、钉钉等。

P2-8: 提供告警回调 handler 示例实现，供 ICMonitor 等组件注册使用。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WecomWebhookHandler:
    """P2-8: 企业微信机器人 Webhook 告警处理器。

    将 ICDecayAlert 等告警消息通过企业微信群机器人推送。

    用法::

        from src.notify.webhook import WecomWebhookHandler
        from src.features.ic_monitor import ICMonitor

        handler = WecomWebhookHandler(url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY")
        monitor = ICMonitor(store_path="data/logs/ic_monitor.json", db_path="data/market.duckdb")
        alerts = monitor.check_decay_alerts(window=20, threshold=0.03, alert_handler=handler)
    """

    def __init__(self, url: str, *, timeout: float = 10.0, mention_all: bool = False) -> None:
        """
        Parameters
        ----------
        url : str
            企业微信群机器人 Webhook 地址。
        timeout : float
            HTTP 请求超时秒数。
        mention_all : bool
            是否 @所有人。
        """
        self.url = str(url)
        self.timeout = float(timeout)
        self.mention_all = bool(mention_all)

    def __call__(self, alert: object) -> bool:
        """发送单条告警消息到企业微信。

        Parameters
        ----------
        alert : ICDecayAlert 或任何实现了 __str__ 的对象

        Returns
        -------
        bool : 发送是否成功
        """
        message = str(alert)
        return self.send_markdown(message)

    def send_markdown(self, content: str) -> bool:
        """以 Markdown 格式发送消息。

        Parameters
        ----------
        content : str
            Markdown 格式的消息正文。

        Returns
        -------
        bool
        """
        payload: dict = {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            },
        }
        if self.mention_all:
            payload["markdown"]["mentioned_list"] = ["@all"]

        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("errcode") != 0:
                logger.warning(
                    "企业微信 Webhook 返回错误: errcode=%s, errmsg=%s",
                    result.get("errcode"),
                    result.get("errmsg", ""),
                )
                return False
            return True
        except requests.exceptions.RequestException as exc:
            logger.error("企业微信 Webhook 请求失败: %s", exc)
            return False
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("企业微信 Webhook 响应解析失败: %s", exc)
            return False

    def send_text(self, content: str, mentioned_list: Optional[list[str]] = None) -> bool:
        """以纯文本格式发送消息。

        Parameters
        ----------
        content : str
            消息正文（最长 2048 字节）。
        mentioned_list : list[str] or None
            要 @的成员 userid 列表；["@all"] 表示 @所有人。

        Returns
        -------
        bool
        """
        payload: dict = {
            "msgtype": "text",
            "text": {
                "content": content,
            },
        }
        if mentioned_list:
            payload["text"]["mentioned_list"] = mentioned_list

        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("errcode") != 0:
                logger.warning(
                    "企业微信 Webhook 返回错误: errcode=%s, errmsg=%s",
                    result.get("errcode"),
                    result.get("errmsg", ""),
                )
                return False
            return True
        except requests.exceptions.RequestException as exc:
            logger.error("企业微信 Webhook 请求失败: %s", exc)
            return False
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("企业微信 Webhook 响应解析失败: %s", exc)
            return False
