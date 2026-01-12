# 股票和ETF连涨分析系统设置说明

## 1. 手动运行分析脚本

如果您无法设置crontab定时任务，可以使用以下方式手动运行分析流程：

### 方法一：直接运行Python脚本
```bash
# 运行完整分析流程（包括数据更新、分析、邮件发送）
/opt/homebrew/Caskroom/miniconda/base/envs/python11/bin/python /Users/wq/Documents/github/learn_llm/agent/tool/chain.py
```

### 方法二：使用Shell脚本
1. 给脚本添加执行权限：
```bash
chmod +x /Users/wq/Documents/github/learn_llm/run_analysis.sh
```

2. 运行脚本：
```bash
./run_analysis.sh
```

## 2. 设置crontab定时任务（推荐）

当您获得系统权限后，可以按照以下步骤设置每天下午14点自动运行分析流程：

### 步骤1：打开crontab编辑器
```bash
crontab -e
```

### 步骤2：添加定时任务
在打开的编辑器中，添加以下行：
```
0 16 * * * /opt/homebrew/Caskroom/miniconda/base/envs/python11/bin/python /Users/wq/Documents/github/learn_llm/agent/tool/chain.py
```

### 步骤3：保存并退出
- 如果使用的是vim编辑器：按`Esc`键，然后输入`:wq`并按`Enter`键
- 如果使用的是nano编辑器：按`Ctrl+O`保存，按`Enter`确认，然后按`Ctrl+X`退出

### 步骤4：验证定时任务设置
```bash
crontab -l
```

## 3. 定时任务解释

`0 14 * * *` 表示：
- 0：分钟（0-59）
- 14：小时（24小时制，14表示下午2点）
- *：日期（1-31，*表示每天）
- *：月份（1-12，*表示每月）
- *：星期几（0-7，0和7都表示星期日，*表示每天）

## 4. 功能说明

当定时任务或手动运行时，系统会执行以下操作：
1. 检查当前是否为交易日
2. 自动更新股票和ETF列表数据
3. 更新股票和ETF的日线数据
4. 分析股票连涨情况（过滤ST股票、科创板、创业板、北京股）
5. 分析ETF连涨情况（过滤总市值不足1亿的ETF）
6. 生成分析结果表格
7. 将分析结果通过邮件发送到指定邮箱

## 5. 故障排除

### 无法运行Python脚本
- 确保conda环境已正确安装：`/opt/homebrew/Caskroom/miniconda/base/envs/python11/bin/python --version`
- 确保脚本路径正确

### 邮件未收到
- 检查脚本运行日志是否有错误信息
- 确保邮箱SMTP设置正确

### 数据更新失败
- 确保网络连接正常
- 检查AKShare库是否已更新到最新版本

## 6. 联系方式

如果遇到问题，请联系系统管理员。
