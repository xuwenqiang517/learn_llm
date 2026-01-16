import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# 发件人信息
sender_email = "since480@163.com"  # 比如 123456@163.com
sender_password = "FP3w2trEpAqPN4x8"    # 不是登录密码，是邮箱的SMTP授权码
smtp_server = "smtp.163.com"         # 邮箱SMTP服务器，不同邮箱不一样
smtp_port = 465                      # SSL端口，一般是465

# 收件人信息
receiver_email = "598570789@qq.com"  # 可以是多个，用列表：["a@qq.com", "b@163.com"]
# 获取当前日期
today_date = datetime.now().strftime("%Y-%m-%d")
email_subject = f"【JDb】选股助手_连涨数据{today_date}"  # 邮件标题

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 分析结果文件路径
BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
ANALYZER_DIR = DATA_DIR / "analyzer"
PICK_DIR = DATA_DIR / "pick"

def generate_html_table_from_csv(csv_path):
    """从CSV文件生成HTML表格"""
    try:
        df = pd.read_csv(csv_path, dtype={'代码': str, '股票代码': str, 'ETF代码': str})

        if 'ETF代码' in df.columns:
            data_type = 'ETF'
            title = 'ETF精选'
        elif '股票代码' in df.columns:
            data_type = '股票'
            title = '股票精选'
        else:
            return None, None, None

        return df, data_type, title

    except Exception as e:
        print(f"读取CSV文件失败 {csv_path}: {e}")
        return None, None, None

def send_analyzer_table():
    today = datetime.now().strftime("%Y%m%d")

    files_to_send = [
        PICK_DIR / f"etf_{today}.csv",
        PICK_DIR / f"stock_{today}.csv"
    ]

    existing_files = []
    for file_path in files_to_send:
        if file_path.exists():
            existing_files.append(file_path)
        else:
            print(f"警告：文件不存在 {file_path}")
    
    if not existing_files:
        print("没有找到要发送的文件")
        return
    
    try:
        # 创建邮件对象
        msg = MIMEMultipart('related')
        msg['From'] = sender_email
        msg['To'] = receiver_email if isinstance(receiver_email, str) else ", ".join(receiver_email)
        msg['Subject'] = email_subject
        
        # 创建邮件HTML正文
        html = """
        <html>
          <head>
            <meta charset='utf-8'>
            <title>选股助手</title>
            <style>
              body { font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; margin: 0 auto; max-width: 1200px; padding: 20px; }
              h1 { color: #2c3e50; text-align: center; }
              h2 { color: #3498db; margin-top: 40px; margin-bottom: 20px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
              table { border-collapse: collapse; width: 100%; margin: 20px 0; }
              th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
              th { background-color: #f2f2f2; font-weight: bold; color: #333; }
              tr:nth-child(even) { background-color: #f9f9f9; }
              tr:hover { background-color: #f5f5f5; }
              .table-container { overflow-x: auto; }
              .footer { margin-top: 50px; color: #7f8c8d; font-size: 14px; text-align: center; }
              .summary { margin: 20px 0; padding: 15px; background-color: #e8f4f8; border-left: 4px solid #3498db; }
            </style>
          </head>
          <body>
            <h1>选股助手 - 精选股票/ETF</h1>
            <div class="summary">
              <p><strong>筛选条件：</strong></p>
              <ul>
                <li>过滤创业板（300/301开头）、科创板（688开头）、北交所（8/92开头）</li>
                <li>过滤ST/*ST股票</li>
                <li>过滤数据不完整的股票（近15天缺失超过3天）</li>
                <li>5日多头排列（MA5 > MA10 > MA20）</li>
                <li>3日涨幅超过15%</li>
                <li>5日涨幅超过20%</li>
              </ul>
            </div>
        """
        
        # 为每个CSV文件生成HTML表格并添加到邮件
        for file_path in existing_files:
            df, data_type, title = generate_html_table_from_csv(file_path)
            if df is not None and data_type and title:
                # 转换为HTML表格
                table_html = df.to_html(index=False, classes='data-table')
                
                html += f"""
                <h2>{title}</h2>
                <div class="table-container">
                  {table_html}
                </div>
                <p>共 {len(df)} 条记录</p>
                """
        
        # 添加邮件底部
        html += """
            <div class="footer">
              <p>此邮件由系统自动生成，请勿回复。</p>
            </div>
          </body>
        </html>
        """
        
        # 添加HTML正文
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        # 连接SMTP服务器并发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"邮件发送成功！已发送 {len(existing_files)} 个HTML表格到 {receiver_email}")
        
    except Exception as e:
        print(f"邮件发送失败：{e}")

def send_email(subject:str,msg_content:str)->None:
    try:
        msg = MIMEMultipart('related')
        msg['From'] = sender_email
        msg['To'] = receiver_email if isinstance(receiver_email, str) else ", ".join(receiver_email)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(msg_content.replace('\n', '<br>'), 'html', 'utf-8'))
        
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"邮件发送成功！已发送到 {receiver_email}")
        
    except Exception as e:
        print(f"邮件发送失败：{e}")

if __name__ == "__main__":
    send_analyzer_table()