# 飞书图片发送配置说明

## 问题说明

当前概念分析图表无法发送到飞书，因为缺少飞书应用的 `app_id` 和 `app_secret` 凭证。这些凭证是上传图片到飞书云空间所必需的。

## 解决方案

### 1. 获取飞书应用凭证

按照以下步骤获取飞书应用的 `app_id` 和 `app_secret`：

1. **登录飞书开放平台**
   - 访问: https://open.feishu.cn/
   - 使用飞书账号登录

2. **创建应用或选择已有应用**
   - 点击"创建应用"或选择已有的应用
   - 选择"自建应用"类型

3. **获取应用凭证**
   - 进入应用详情页
   - 在"凭证与基础信息"页面可以找到：
     - App ID
     - App Secret

4. **配置应用权限**
   - 在"权限管理"中添加以下权限：
     - `im:message` (发送消息)
     - `drive:drive:readonly` (读取云空间)
     - `drive:file:readonly` (读取文件)
   - 发布应用或申请权限审核

5. **配置机器人**
   - 在"机器人"页面添加自定义机器人
   - 获取 Webhook URL（当前已配置）

### 2. 配置凭证到代码

有两种方式配置凭证：

#### 方式1: 直接修改代码（测试用）

在调用 `search_concepts` 函数时传入凭证：

```python
from agent.stock_searh_tool import search_concepts

result = search_concepts(
    days=5,
    market="all",
    save_result=True,
    use_cache=False,
    include_kc=False,
    include_cy=False,
    auto_update_cache=True,
    send_to_feishu=True,
    top_n=20,
    feishu_app_id="your_app_id_here",  # 替换为实际的 app_id
    feishu_app_secret="your_app_secret_here"  # 替换为实际的 app_secret
)
```

#### 方式2: 使用环境变量（推荐）

创建 `.env` 文件（推荐方式）：

```bash
# 飞书应用配置
FEISHU_APP_ID=your_app_id_here
FEISHU_APP_SECRET=your_app_secret_here
```

然后修改代码读取环境变量：

```python
import os
from dotenv import load_dotenv

load_dotenv()

feishu_app_id = os.getenv("FEISHU_APP_ID")
feishu_app_secret = os.getenv("FEISHU_APP_SECRET")
```

### 3. 测试功能

运行测试脚本：

```bash
conda activate py311
python test_concept_analysis_feishu.py
```

### 4. 验证结果

配置凭证后，再次运行概念分析，您应该能够在飞书中收到：
1. 文本消息（包含分析摘要）
2. 图片附件（概念分析图表）

## 当前状态

- ✅ 代码已修复，支持传递 `app_id` 和 `app_secret` 参数
- ✅ 图片上传和发送功能已实现
- ⚠️  需要配置有效的飞书应用凭证才能发送图片
- ✅ 文本消息可以正常发送（不需要凭证）

## 注意事项

1. **安全性**: 不要将 `app_id` 和 `app_secret` 提交到代码仓库
2. **权限**: 确保应用有足够的权限发送消息和上传图片
3. **测试**: 先在测试群中测试，确认功能正常后再在生产环境使用
4. **Webhook**: 当前 Webhook URL 已配置，无需修改

## 技术细节

图片发送流程：
1. 使用 `app_id` 和 `app_secret` 获取 `tenant_access_token`
2. 使用 `tenant_access_token` 上传图片到飞书云空间
3. 获取图片的 `image_key`
4. 使用 `image_key` 发送图片消息到飞书群聊

如果没有配置凭证，系统会：
- 记录日志："未配置 app_id 和 app_secret，仅发送文本消息"
- 仍然发送文本消息到飞书
- 不会发送图片附件
