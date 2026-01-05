"""
测试概念分析功能并发送到飞书
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.stock_searh_tool import search_concepts

def test_concept_analysis():
    """测试概念分析功能"""
    print("=" * 70)
    print("测试概念分析功能")
    print("=" * 70)
    
    # 配置飞书应用凭证（需要用户提供）
    # 请替换为实际的飞书应用 ID 和 Secret
    FEISHU_APP_ID = None  # 替换为实际的 app_id
    FEISHU_APP_SECRET = None  # 替换为实际的 app_secret
    
    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        print("\n⚠️  未配置飞书应用凭证")
        print("请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET 变量")
        print("\n如何获取飞书应用凭证:")
        print("1. 登录飞书开放平台: https://open.feishu.cn/")
        print("2. 创建应用或选择已有应用")
        print("3. 在应用详情页获取 App ID 和 App Secret")
        print("4. 确保应用有发送消息和上传图片的权限")
        print("\n如果暂时不发送图片，可以继续测试分析功能")
    
    print("\n开始概念分析...")
    
    result = search_concepts(
        days=5,
        market="all",
        save_result=True,
        use_cache=False,
        include_kc=False,
        include_cy=False,
        auto_update_cache=True,
        send_to_feishu=True,  # 发送到飞书
        top_n=20,
        feishu_app_id=FEISHU_APP_ID,
        feishu_app_secret=FEISHU_APP_SECRET
    )
    
    print(f"\n查询结果: {result.get('message', 'N/A')}")
    
    if result.get("success"):
        print(f"✅ 概念分析成功")
        print(f"📊 图表路径: {result.get('chart_path', 'N/A')}")
        
        data = result.get("data", {})
        if data and isinstance(data, dict):
            concepts = data.get('concepts', [])
            if concepts:
                print(f"\n前5个概念:")
                for i, concept in enumerate(concepts[:5], 1):
                    print(f"  {i}. {concept.get('name', 'N/A')}: {concept.get('total_avg_change', 'N/A')}")
    else:
        print("❌ 概念分析失败")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_concept_analysis()
