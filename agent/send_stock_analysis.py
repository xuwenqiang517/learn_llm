"""
飞书机器人消息发送工具
将股票分析结果发送到飞书群聊
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeishuBot:
    """飞书群机器人客户端"""

    def __init__(self, webhook_url: str, app_id: Optional[str] = None, app_secret: Optional[str] = None):
        """
        初始化飞书机器人

        Args:
            webhook_url: 飞书机器人 Webhook URL
            app_id: 飞书应用 ID（用于图片上传功能）
            app_secret: 飞书应用 Secret（用于图片上传功能）
        """
        self.webhook_url = webhook_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.tenant_access_token = None
        self.session = requests.Session()

    def send_text(self, text: str) -> bool:
        """
        发送文本消息

        Args:
            text: 文本内容

        Returns:
            是否发送成功
        """
        payload = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }
        return self._send(payload)

    def send_markdown(self, title: str, text: str) -> bool:
        """
        发送富文本消息（飞书机器人仅支持text类型）

        Args:
            title: 标题（作为文本前缀）
            text: 文本内容

        Returns:
            是否发送成功
        """
        full_text = f"## {title}\n\n{text}"
        return self.send_text(full_text)

    def send_image(self, image_key: str) -> bool:
        """
        发送图片消息

        Args:
            image_key: 图片的 image_key（通过上传图片获得）

        Returns:
            是否发送成功
        """
        payload = {
            "msg_type": "image",
            "content": {
                "image_key": image_key
            }
        }
        return self._send(payload)

    def upload_image(self, image_path: str) -> Optional[str]:
        """
        上传图片到飞书云空间

        Args:
            image_path: 图片文件路径

        Returns:
            image_key，上传失败返回 None
        """
        if not self.app_id or not self.app_secret:
            logger.error("未配置 app_id 和 app_secret，无法上传图片")
            return None

        try:
            image_file = Path(image_path)
            if not image_file.exists():
                logger.error(f"图片文件不存在: {image_path}")
                return None

            if not self.tenant_access_token:
                self.tenant_access_token = self._get_tenant_access_token()
                if not self.tenant_access_token:
                    return None

            upload_url = f"https://open.feishu.cn/open-apis/drive/v1/medias/upload_all"
            headers = {
                "Authorization": f"Bearer {self.tenant_access_token}"
            }

            files = {
                "file": (image_file.name, open(image_file, "rb"), "image/png")
            }
            data = {
                "file_type": "image",
                "parent_type": "explorer"
            }

            response = self.session.post(upload_url, headers=headers, files=files, data=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            if result.get("code") == 0:
                image_key = result.get("data", {}).get("file", {}).get("file_token")
                logger.info(f"图片上传成功，image_key: {image_key}")
                return image_key
            else:
                logger.error(f"图片上传失败: {result.get('msg')}")
                return None

        except requests.RequestException as e:
            logger.error(f"上传图片请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"上传图片异常: {e}")
            return None

    def _get_tenant_access_token(self) -> Optional[str]:
        """
        获取 tenant_access_token

        Returns:
            tenant_access_token，失败返回 None
        """
        try:
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            payload = {
                "app_id": self.app_id,
                "app_secret": self.app_secret
            }
            headers = {"Content-Type": "application/json; charset=utf-8"}

            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            if result.get("code") == 0:
                token = result.get("tenant_access_token")
                logger.info("获取 tenant_access_token 成功")
                return token
            else:
                logger.error(f"获取 tenant_access_token 失败: {result.get('msg')}")
                return None

        except requests.RequestException as e:
            logger.error(f"获取 tenant_access_token 请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"获取 tenant_access_token 异常: {e}")
            return None

    def _send(self, payload: dict) -> bool:
        """
        发送消息

        Args:
            payload: 消息载荷

        Returns:
            是否发送成功
        """
        try:
            headers = {"Content-Type": "application/json; charset=utf-8"}
            response = self.session.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            if result.get("code") == 0:
                logger.info("消息发送成功")
                return True
            else:
                logger.error(f"消息发送失败: {result.get('msg')}")
                return False

        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            return False
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            return False


def get_latest_analysis_file(temp_dir: str = "/Users/JDb/Desktop/github/learn_llm/.temp") -> Optional[Path]:
    """
    获取最新的股票分析文件（从 output/analysis/ 目录）

    Args:
        temp_dir: .temp 目录路径

    Returns:
        最新分析文件的 Path 对象，如果没有找到返回 None
    """
    analysis_dir = Path(temp_dir) / "output" / "analysis"
    if not analysis_dir.exists():
        logger.error(f"目录不存在: {analysis_dir}")
        return None

    analysis_files = list(analysis_dir.glob("stock_analysis_*.md"))
    if not analysis_files:
        logger.error("未找到股票分析文件")
        return None

    latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"找到最新分析文件: {latest_file}")
    return latest_file


def get_latest_table_file(temp_dir: str = "/Users/JDb/Desktop/github/learn_llm/.temp") -> Optional[Path]:
    """
    获取最新的股票表格文件（从 output/tools/ 目录）

    Args:
        temp_dir: .temp 目录路径

    Returns:
        最新表格文件的 Path 对象，如果没有找到返回 None
    """
    tools_dir = Path(temp_dir) / "output" / "tools"
    if not tools_dir.exists():
        logger.error(f"目录不存在: {tools_dir}")
        return None

    table_files = list(tools_dir.glob("rising_stocks_*.md")) + list(tools_dir.glob("stock_table_*.md"))
    if not table_files:
        logger.warning("未找到股票表格文件")
        return None

    latest_file = max(table_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"找到最新表格文件: {latest_file}")
    return latest_file


def get_latest_concept_chart(temp_dir: str = "/Users/JDb/Desktop/github/learn_llm/.temp") -> Optional[Path]:
    """
    获取最新的概念分析图表文件（从 output/tools/ 目录）

    Args:
        temp_dir: .temp 目录路径

    Returns:
        最新概念分析图表文件的 Path 对象，如果没有找到返回 None
    """
    tools_dir = Path(temp_dir) / "output" / "tools"
    if not tools_dir.exists():
        logger.error(f"目录不存在: {tools_dir}")
        return None

    chart_files = list(tools_dir.glob("concept_analysis_*.png"))
    if not chart_files:
        logger.warning("未找到概念分析图表文件")
        return None

    latest_file = max(chart_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"找到最新概念分析图表: {latest_file}")
    return latest_file


def format_analysis_for_feishu(file_path: Path) -> tuple[str, str]:
    """
    格式化分析文件为飞书消息格式

    Args:
        file_path: 分析文件路径

    Returns:
        (标题, Markdown内容)
    """
    content = file_path.read_text(encoding="utf-8")

    today = datetime.now().strftime("%Y年%m月%d日")
    title = f"📈 股票分析报告 - {today}"

    header = f"## {title}\n\n"
    
    # 从文件内容中提取模型版本信息，并从内容中移除
    model_info = ""
    lines = content.split('\n')
    content_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        
        # 检测到分隔线且下一行包含模型信息
        if line.strip() == '---' and i + 1 < len(lines):
            next_line = lines[i + 1]
            if '分析模型' in next_line or '生成时间' in next_line:
                # 提取模型信息
                model_info += line + '\n'
                skip_next = True
                # 继续提取后续的模型信息行
                for j in range(i + 1, len(lines)):
                    if j < len(lines) and ('分析模型' in lines[j] or '生成时间' in lines[j]):
                        model_info += lines[j] + '\n'
                    else:
                        break
                continue
        
        content_lines.append(line)
    
    content = '\n'.join(content_lines)
    
    if model_info:
        footer = f"\n\n{model_info}"
    else:
        footer = f"\n\n---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

    formatted_content = header + content + footer

    max_length = 40000
    if len(formatted_content) > max_length:
        logger.warning(f"内容过长（{len(formatted_content)}字符），将进行截断")
        formatted_content = formatted_content[:max_length] + "\n\n...内容过长，已截断"

    return title, formatted_content


def send_latest_analysis(
    webhook_url: str = "https://open.feishu.cn/open-apis/bot/v2/hook/c8278f54-8e18-4edc-97bd-0c0abc3ab17f",
    temp_dir: str = "/Users/JDb/Desktop/github/learn_llm/.temp",
    include_table: bool = True
) -> bool:
    """
    发送最新的股票分析结果到飞书

    Args:
        webhook_url: 飞书机器人 Webhook URL
        temp_dir: .temp 目录路径
        include_table: 是否同时发送表格数据

    Returns:
        是否发送成功
    """
    try:
        logger.info("开始发送股票分析结果到飞书...")

        latest_file = get_latest_analysis_file(temp_dir)
        if not latest_file:
            logger.error("未找到分析文件")
            return False

        title, content = format_analysis_for_feishu(latest_file)

        bot = FeishuBot(webhook_url)

        if not bot.send_markdown(title, content):
            logger.error("分析文件发送失败")
            return False

        logger.info(f"已发送分析文件: {latest_file.name}")

        if include_table:
            table_file = get_latest_table_file(temp_dir)
            if table_file:
                logger.info(f"发送表格文件: {table_file.name}")
                table_content = table_file.read_text(encoding="utf-8")
                table_title = f"📊 股票数据表格 - {datetime.now().strftime('%Y年%m月%d日')}"
                bot.send_markdown(table_title, table_content)
                logger.info("表格发送成功")
            else:
                logger.warning("未找到表格文件，仅发送分析内容")

        return True
    except Exception as e:
        logger.error(f"发送飞书消息异常: {e}")
        return False


def send_all_results(
    webhook_url: str = "https://open.feishu.cn/open-apis/bot/v2/hook/c8278f54-8e18-4edc-97bd-0c0abc3ab17f",
    temp_dir: str = "/Users/JDb/Desktop/github/learn_llm/.temp"
) -> bool:
    """
    发送所有股票分析结果（分析报告 + 表格数据）

    Args:
        webhook_url: 飞书机器人 Webhook URL
        temp_dir: .temp 目录路径

    Returns:
        是否发送成功
    """
    return send_latest_analysis(webhook_url=webhook_url, temp_dir=temp_dir, include_table=True)


def send_analysis_file(
    file_path: str,
    webhook_url: str = "https://open.feishu.cn/open-apis/bot/v2/hook/c8278f54-8e18-4edc-97bd-0c0abc3ab17f"
) -> bool:
    """
    发送指定的分析文件到飞书

    Args:
        file_path: 分析文件路径
        webhook_url: 飞书机器人 Webhook URL

    Returns:
        是否发送成功
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"文件不存在: {file_path}")
        return False

    logger.info(f"发送文件: {file_path}")

    title, content = format_analysis_for_feishu(path)

    bot = FeishuBot(webhook_url)

    return bot.send_markdown(title, content)


def send_concept_analysis_chart(
    webhook_url: str = "https://open.feishu.cn/open-apis/bot/v2/hook/c8278f54-8e18-4edc-97bd-0c0abc3ab17f",
    temp_dir: str = "/Users/JDb/Desktop/github/learn_llm/.temp",
    app_id: Optional[str] = None,
    app_secret: Optional[str] = None
) -> bool:
    """
    发送最新的概念分析图表到飞书

    Args:
        webhook_url: 飞书机器人 Webhook URL
        temp_dir: .temp 目录路径
        app_id: 飞书应用 ID（用于图片上传功能）
        app_secret: 飞书应用 Secret（用于图片上传功能）

    Returns:
        是否发送成功
    """
    try:
        logger.info("开始发送概念分析图表到飞书...")

        chart_file = get_latest_concept_chart(temp_dir)
        if not chart_file:
            logger.error("未找到概念分析图表文件")
            return False

        today = datetime.now().strftime("%Y年%m月%d日")
        title = f"📊 大盘概念分析 - {today}"
        
        # 构建消息内容（不包含 ## 标题，因为 send_markdown 会自动添加）
        message = f"""📈 **概念趋势分析**

已生成最新的概念分析图表，包含以下内容：
- Top 20 概念的5日累计涨跌幅
- 每个概念对应股票的详细数据
- 最近5个交易日的价格变化趋势

**图表文件**: {chart_file.name}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*请查看附件中的概念分析图表*
"""

        bot = FeishuBot(webhook_url, app_id, app_secret)

        if not bot.send_markdown(title, message):
            logger.error("概念分析图表发送失败")
            return False

        # 尝试上传并发送图片
        if app_id and app_secret:
            try:
                image_key = bot.upload_image(str(chart_file))
                if image_key:
                    if bot.send_image(image_key):
                        logger.info(f"已发送概念分析图片: {chart_file.name}")
                    else:
                        logger.warning("图片发送失败，仅发送了文本消息")
                else:
                    logger.warning("图片上传失败，仅发送了文本消息")
            except Exception as e:
                logger.warning(f"图片发送异常: {e}，仅发送了文本消息")
        else:
            logger.info("未配置 app_id 和 app_secret，仅发送文本消息")

        logger.info(f"已发送概念分析图表: {chart_file.name}")
        return True

    except Exception as e:
        logger.error(f"发送概念分析图表异常: {e}")
        return False


def main():
    """主函数 - 演示发送功能"""
    print("=" * 70)
    print("飞书消息发送工具")
    print("=" * 70)

    print("\n1. 发送最新的股票分析报告和表格数据...")
    success = send_all_results()

    if success:
        print("\n✅ 发送成功！")
    else:
        print("\n❌ 发送失败")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
