#!/usr/bin/env python3
"""
股票分析代理测试脚本
验证日志、调试模式、工具调用和KC/CY参数过滤功能
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from agent.stock_agent import run_stock_analysis, search_rising_stocks_tool
from agent.stock_searh_tool import get_rising_stocks


def test_tool_parameters():
    """测试工具参数是否正确传递"""
    print("\n" + "="*70)
    print("测试1: 验证KC/CY参数默认值为False")
    print("="*70)
    
    # 测试1: 验证默认参数 - 通过访问inner_function
    print("\n1.1 测试搜索工具默认参数:")
    tool_obj = search_rising_stocks_tool
    # 获取原始函数
    func = getattr(tool_obj, 'func', tool_obj)
    import inspect
    sig = inspect.signature(func)
    params = sig.parameters
    print(f"   include_kc 默认值: {params['include_kc'].default}")
    print(f"   include_cy 默认值: {params['include_cy'].default}")
    assert params['include_kc'].default == False, "include_kc 应该默认为 False"
    assert params['include_cy'].default == False, "include_cy 应该默认为 False"
    print("   ✅ 参数默认值验证通过")
    
    print("\n1.2 测试get_rising_stocks函数默认参数:")
    sig = inspect.signature(get_rising_stocks)
    params = sig.parameters
    print(f"   include_kc 默认值: {params['include_kc'].default}")
    print(f"   include_cy 默认值: {params['include_cy'].default}")
    assert params['include_kc'].default == False, "include_kc 应该默认为 False"
    assert params['include_cy'].default == False, "include_cy 应该默认为 False"
    print("   ✅ 函数参数默认值验证通过")


def test_tool_filtering():
    """测试KC/CY过滤功能"""
    print("\n" + "="*70)
    print("测试2: 验证KC/CY过滤功能")
    print("="*70)
    
    print("\n2.1 测试包含KC/CY (include_kc=True, include_cy=True):")
    result_with_kc_cy = get_rising_stocks(days=3, min_increase=5.0, include_kc=True, include_cy=True)
    print(f"   包含KC/CY的股票数量: {len(result_with_kc_cy)}")
    if len(result_with_kc_cy) > 0:
        kc_stocks = result_with_kc_cy[result_with_kc_cy['code'].str.startswith('68')]
        cy_stocks = result_with_kc_cy[result_with_kc_cy['code'].str.startswith('3')]
        print(f"   其中科创板股票: {len(kc_stocks)} 只")
        print(f"   其中创业板股票: {len(cy_stocks)} 只")
    
    print("\n2.2 测试不包含KC/CY (include_kc=False, include_cy=False):")
    result_without_kc_cy = get_rising_stocks(days=3, min_increase=5.0, include_kc=False, include_cy=False)
    print(f"   不包含KC/CY的股票数量: {len(result_without_kc_cy)}")
    if len(result_without_kc_cy) > 0:
        kc_stocks = result_without_kc_cy[result_without_kc_cy['code'].str.startswith('68')]
        cy_stocks = result_without_kc_cy[result_without_kc_cy['code'].str.startswith('3')]
        print(f"   其中科创板股票: {len(kc_stocks)} 只")
        print(f"   其中创业板股票: {len(cy_stocks)} 只")
        assert len(kc_stocks) == 0, "不应该包含科创板股票"
        assert len(cy_stocks) == 0, "不应该包含创业板股票"
        print("   ✅ KC/CY过滤功能验证通过")


def test_agent_debug_mode():
    """测试代理调试模式"""
    print("\n" + "="*70)
    print("测试3: 验证代理调试模式")
    print("="*70)
    
    print("\n3.1 运行带调试模式的股票分析:")
    user_query = "分析最近连续3天上涨的股票，要求排除科创板和创业板"
    print(f"   用户查询: {user_query}")
    
    # 使用调试模式运行
    result = run_stock_analysis(user_query, debug=True)
    
    print("\n3.2 分析结果:")
    print(f"   结果长度: {len(result)} 字符")
    print(f"   是否包含有效内容: {'是' if result != '未获取到有效响应' else '否'}")
    
    if result != "未获取到有效响应":
        print("   ✅ 代理成功返回分析结果")
        # 检查结果是否包含中文分析
        has_chinese_analysis = any(keyword in result for keyword in ['涨幅', '行业', '概念', '股票', '分析'])
        if has_chinese_analysis:
            print("   ✅ 结果包含中文分析内容")
        else:
            print("   ⚠️ 结果可能不包含预期的分析内容")
    else:
        print("   ❌ 代理返回空结果，需要进一步调试")
    
    return result


def test_agent_normal_mode():
    """测试代理普通模式"""
    print("\n" + "="*70)
    print("测试4: 验证代理普通模式")
    print("="*70)
    
    print("\n4.1 运行普通模式的股票分析:")
    user_query = "找出最近3天涨幅超过5%的股票"
    print(f"   用户查询: {user_query}")
    
    # 使用普通模式运行
    result = run_stock_analysis(user_query, debug=False)
    
    print("\n4.2 分析结果:")
    print(f"   结果长度: {len(result)} 字符")
    print(f"   是否包含有效内容: {'是' if result != '未获取到有效响应' else '否'}")
    
    if result != "未获取到有效响应":
        print("   ✅ 普通模式运行成功")
    else:
        print("   ⚠️ 普通模式返回空结果")
    
    return result


def main():
    """主测试函数"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "    股票分析代理系统功能测试".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    try:
        # 测试1: 验证参数默认值
        test_tool_parameters()
        
        # 测试2: 验证过滤功能
        test_tool_filtering()
        
        # 测试3: 验证调试模式
        debug_result = test_agent_debug_mode()
        
        # 测试4: 验证普通模式
        normal_result = test_agent_normal_mode()
        
        # 总结
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        print(f"\n✅ 所有测试完成!")
        print(f"\n调试模式结果预览:")
        print(f"   {debug_result[:500] if debug_result != '未获取到有效响应' else debug_result}...")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.exception("测试过程中发生异常")
        sys.exit(1)


if __name__ == "__main__":
    main()
