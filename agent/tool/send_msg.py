# 发件人信息
sender_email = "since480@163.com"  # 比如 123456@163.com
sender_password = "FP3w2trEpAqPN4x8"    # 不是登录密码，是邮箱的SMTP授权码
smtp_server = "smtp.163.com"         # 邮箱SMTP服务器，不同邮箱不一样
smtp_port = 465                      # SSL端口，一般是465

# 收件人信息
receiver_email = "598570789@qq.com"  # 可以是多个，用列表：["a@qq.com", "b@163.com"]
email_subject = "【测试】发送分析表格到邮箱"  # 邮件标题


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import sys
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 确保使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']  # macOS和Linux的中文字体选项
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 分析结果文件路径
BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
ANALYZER_DIR = DATA_DIR / "analyzer"

def generate_table_image_from_csv(csv_path):
    """从CSV文件生成表格图片"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 确定是股票还是ETF数据
        if '股票代码' in df.columns:
            data_type = '股票'
            title = '股票连涨分析表格'
        elif 'ETF代码' in df.columns:
            data_type = 'ETF'
            title = 'ETF连涨分析表格'
        else:
            return None, None
        
        # 显示全部内容
        show_more = ""
        
        # 创建表格
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')  # 隐藏坐标轴
        
        # 创建表格
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        
        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)  # 调整表格大小
        
        # 设置表格标题
        plt.title(f"{title}{show_more}", fontsize=14, pad=20)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存表格到内存
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # 清理
        plt.close()
        buffer.close()
        
        return image_base64, data_type
        
    except Exception as e:
        print(f"生成表格图片失败 {csv_path}: {e}")
        return None, None

def send_analyzer_table():
    # 邮件附件文件列表（ETF在前，股票在后）
    files_to_send = [
        ANALYZER_DIR / "etf_rising.csv",
        ANALYZER_DIR / "stock_rising.csv"
    ]
    
    # 检查文件是否存在
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
            <title>分析结果</title>
            <style>
              body { font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; }
              h1 { color: #2c3e50; }
              h2 { color: #3498db; margin-top: 30px; }
              .image-container { margin: 20px 0; text-align: center; }
              img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; background-color: white; }
              .footer { margin-top: 50px; color: #7f8c8d; font-size: 14px; }
            </style>
          </head>
          <body>
            <h1>股票/ETF连涨分析结果</h1>
            <p>您好！以下是最新的分析表格：</p>
        """
        
        # 为每个CSV文件生成表格图片并添加到邮件
        for file_path in existing_files:
            image_base64, data_type = generate_table_image_from_csv(file_path)
            if image_base64 and data_type:
                html += f"""
                <h2>{data_type}连涨分析表格</h2>
                <div class="image-container">
                  <img src="data:image/png;base64,{image_base64}" alt="{data_type}分析表格">
                </div>
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
        
        print(f"邮件发送成功！已发送 {len(existing_files)} 个表格图片到 {receiver_email}")
        
    except Exception as e:
        print(f"邮件发送失败：{e}")

if __name__ == "__main__":
    send_analyzer_table()