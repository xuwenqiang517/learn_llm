"""
真实数据文件技术指标验证测试

基于 load_base_data.py 产出的真实数据文件
验证所有计算加工字段的准确性，发现并修复逻辑漏洞
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, '/Users/wq/Documents/github/learn_llm')

from utils.data_path_util import get_stock_data_dir, get_etf_data_dir


def get_sample_files(n: int = 20) -> list:
    """获取样本文件进行测试"""
    stock_dir = get_stock_data_dir()
    stock_files = list(stock_dir.glob("*.csv"))[:n]
    return stock_files


def manual_ma(close: pd.Series, window: int) -> pd.Series:
    """手动计算移动平均线"""
    return close.rolling(window=window).mean().round(2)


def manual_cumulative_returns(pct_change: pd.Series, window: int) -> pd.Series:
    """手动计算累计涨幅"""
    return pct_change.rolling(window=window).sum().round(2)


def manual_consecutive_rise(pct_change: pd.Series) -> pd.Series:
    """手动计算连涨天数"""
    is_rising = (pct_change > 0).astype(int)
    consecutive = is_rising.cumsum() - is_rising.cumsum().where(is_rising == 0).ffill().fillna(0).astype(int)
    return consecutive


def verify_single_file(file_path: Path) -> dict:
    """验证单个文件的所有计算字段"""
    result = {
        'file': file_path.name,
        'passed': True,
        'errors': []
    }
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', parse_dates=['日期'])
        if df.empty:
            result['passed'] = False
            result['errors'].append('文件为空')
            return result
        
        if len(df) < 30:
            result['passed'] = False
            result['errors'].append(f'数据行数不足: {len(df)}')
            return result
        
        close = df['收盘']
        pct_change = df['涨跌幅']
        high = df['最高']
        low = df['最低']
        
        # 1. 验证MA5
        ma5_actual = df['MA5']
        ma5_expected = manual_ma(close, 5)
        ma5_diff = (ma5_actual.fillna(0) - ma5_expected.fillna(0)).abs().sum()
        if ma5_diff > 0.01:
            result['passed'] = False
            result['errors'].append(f'MA5差异: {ma5_diff:.2f}')
        
        # 2. 验证MA10
        ma10_actual = df['MA10']
        ma10_expected = manual_ma(close, 10)
        ma10_diff = (ma10_actual.fillna(0) - ma10_expected.fillna(0)).abs().sum()
        if ma10_diff > 0.01:
            result['passed'] = False
            result['errors'].append(f'MA10差异: {ma10_diff:.2f}')
        
        # 3. 验证MA20
        ma20_actual = df['MA20']
        ma20_expected = manual_ma(close, 20)
        ma20_diff = (ma20_actual.fillna(0) - ma20_expected.fillna(0)).abs().sum()
        if ma20_diff > 0.01:
            result['passed'] = False
            result['errors'].append(f'MA20差异: {ma20_diff:.2f}')
        
        # 4. 验证3日涨幅
        d3_actual = df['3日涨幅']
        d3_expected = manual_cumulative_returns(pct_change, 3)
        d3_diff = (d3_actual.fillna(0) - d3_expected.fillna(0)).abs().sum()
        if d3_diff > 0.01:
            result['passed'] = False
            result['errors'].append(f'3日涨幅差异: {d3_diff:.2f}')
        
        # 5. 验证5日涨幅
        d5_actual = df['5日涨幅']
        d5_expected = manual_cumulative_returns(pct_change, 5)
        d5_diff = (d5_actual.fillna(0) - d5_expected.fillna(0)).abs().sum()
        if d5_diff > 0.01:
            result['passed'] = False
            result['errors'].append(f'5日涨幅差异: {d5_diff:.2f}')
        
        # 6. 验证连涨天数
        consec_actual = df['连涨天数']
        consec_expected = manual_consecutive_rise(pct_change)
        match_count = (consec_actual.reset_index(drop=True) == consec_expected.reset_index(drop=True)).sum()
        total_count = len(consec_actual)
        if match_count != total_count:
            result['passed'] = False
            mismatch_rate = (total_count - match_count) / total_count * 100
            result['errors'].append(f'连涨天数不匹配: {total_count - match_count}/{total_count} ({mismatch_rate:.1f}%)')
            
            mismatched_indices = consec_actual[consec_actual != consec_expected].index.tolist()
            if mismatched_indices:
                sample_indices = mismatched_indices[:5]
                sample_info = []
                for idx in sample_indices:
                    if idx < len(pct_change):
                        sample_info.append(f'{idx}(涨跌幅={pct_change.iloc[idx]:.2f}%, 实际={consec_actual.iloc[idx]}, 期望={consec_expected.iloc[idx]})')
                result['errors'].append(f'示例: {sample_info}')
        
        # 7. 验证连涨天数逻辑正确性（涨了应该>0，跌了应该=0）
        for idx in range(len(df)):
            pct = pct_change.iloc[idx] if idx < len(pct_change) else 0
            consec = consec_actual.iloc[idx] if idx < len(consec_actual) else 0
            
            if pd.notna(pct) and pd.notna(consec):
                if pct > 0 and consec <= 0:
                    result['passed'] = False
                    result['errors'].append(f'连涨天数逻辑错误(涨了应该>0): idx={idx}, 涨跌幅={pct:.2f}%, 连涨天数={consec}')
                elif pct < 0 and consec != 0:
                    result['passed'] = False
                    result['errors'].append(f'连涨天数逻辑错误(跌了应该=0): idx={idx}, 涨跌幅={pct:.2f}%, 连涨天数={consec}')
        
    except Exception as e:
        result['passed'] = False
        result['errors'].append(f'异常: {str(e)}')
    
    return result


def verify_all_files(max_files: int = 50) -> dict:
    """验证所有数据文件"""
    print("="*70)
    print("真实数据文件技术指标验证")
    print("="*70)
    
    stock_dir = get_stock_data_dir()
    stock_files = list(stock_dir.glob("*.csv"))[:max_files]
    
    print(f"\n验证 {len(stock_files)} 个股票数据文件...")
    
    all_passed = True
    total_errors = 0
    error_summary = {}
    
    for file_path in stock_files:
        result = verify_single_file(file_path)
        if not result['passed']:
            all_passed = False
            total_errors += len(result['errors'])
            print(f"\n❌ {result['file']}")
            for error in result['errors'][:3]:
                print(f"   {error}")
                if '连涨天数' in error or 'MA' in error:
                    key = error.split('差异')[0].strip() if '差异' in error else error.split('(')[0].strip()
                    error_summary[key] = error_summary.get(key, 0) + 1
    
    print("\n" + "="*70)
    print("验证结果汇总")
    print("="*70)
    print(f"验证文件数: {len(stock_files)}")
    print(f"全部通过: {'是' if all_passed else '否'}")
    
    if error_summary:
        print("\n错误类型统计:")
        for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}次")
    
    return {
        'all_passed': all_passed,
        'total_errors': total_errors,
        'error_summary': error_summary
    }


def find_consecutive_rise_bugs(max_files: int = 50) -> list:
    """专门查找连涨天数bug"""
    print("\n" + "="*70)
    print("连涨天数专项检查")
    print("="*70)
    
    stock_dir = get_stock_data_dir()
    stock_files = list(stock_dir.glob("*.csv"))[:max_files]
    
    bug_files = []
    
    for file_path in stock_files:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        if '连涨天数' not in df.columns:
            continue
        
        pct_change = df['涨跌幅']
        consecutive = df['连涨天数']
        
        bugs = []
        
        for idx in range(len(df)):
            if idx >= len(pct_change) or idx >= len(consecutive):
                continue
                
            pct = pct_change.iloc[idx]
            consec = consecutive.iloc[idx]
            
            if pd.isna(pct) or pd.isna(consec):
                continue
            
            # 检查逻辑错误
            if pct > 0 and consec <= 0:
                bugs.append(f'idx={idx}: 涨{pct:.2f}% 但连涨天数={consec}')
            elif pct < 0 and consec != 0:
                bugs.append(f'idx={idx}: 跌{pct:.2f}% 但连涨天数={consec}')
        
        if bugs:
            bug_files.append({
                'file': file_path.name,
                'bugs': bugs[:10]
            })
    
    if bug_files:
        print(f"\n发现 {len(bug_files)} 个文件存在连涨天数bug:")
        for bf in bug_files[:10]:
            print(f"\n📁 {bf['file']}")
            for bug in bf['bugs'][:5]:
                print(f"   {bug}")
        
        if len(bug_files) > 10:
            print(f"\n... 还有 {len(bug_files) - 10} 个文件存在问题")
    else:
        print(f"\n✅ 没有发现连涨天数bug")
    
    return bug_files


def check_vol_ma_indicators(max_files: int = 30) -> list:
    """检查成交量MA指标"""
    print("\n" + "="*70)
    print("成交量MA指标检查")
    print("="*70)
    
    stock_dir = get_stock_data_dir()
    stock_files = list(stock_dir.glob("*.csv"))[:max_files]
    
    error_files = []
    
    for file_path in stock_files:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        if '成交量' not in df.columns:
            continue
        
        volume = df['成交量']
        
        for window in [5, 10, 20]:
            col_name = f'VOL_MA{window}'
            if col_name not in df.columns:
                if len(df) >= window:
                    error_files.append({
                        'file': file_path.name,
                        'issue': f'缺少 {col_name} 列'
                    })
                continue
            
            actual = df[col_name]
            expected = volume.rolling(window=window).mean().round(2)
            diff = (actual.fillna(0) - expected.fillna(0)).abs().sum()
            
            if diff > 0.01:
                error_files.append({
                    'file': file_path.name,
                    'issue': f'{col_name}差异={diff:.2f}'
                })
    
    if error_files:
        print(f"\n发现 {len(error_files)} 个文件存在成交量MA问题:")
        for ef in error_files[:10]:
            print(f"   {ef['file']}: {ef['issue']}")
    else:
        print(f"\n✅ 成交量MA指标检查通过")
    
    return error_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='真实数据文件技术指标验证')
    parser.add_argument('--files', type=int, default=100, help='验证文件数量')
    args = parser.parse_args()
    
    print("\n开始真实数据文件验证...\n")
    
    result1 = verify_all_files(args.files)
    bug_files = find_consecutive_rise_bugs(args.files)
    vol_errors = check_vol_ma_indicators(args.files)
    
    print("\n" + "="*70)
    print("最终结论")
    print("="*70)
    
    if result1['all_passed'] and not bug_files and not vol_errors:
        print(f"🎉 全部验证通过！共验证 {args.files} 个文件，数据完全正确。")
        exit(0)
    else:
        print("⚠️ 存在问题需要修复:")
        if not result1['all_passed']:
            print(f"  - 总体验证失败: {result1['total_errors']} 个错误")
        if bug_files:
            print(f"  - 连涨天数bug: {len(bug_files)} 个文件")
        if vol_errors:
            print(f"  - 成交量MA问题: {len(vol_errors)} 个文件")
        
        print("\n需要修复 load_base_data.py 中的逻辑...")
        exit(1)
