from datetime import datetime
from pathlib import Path
from dataclasses import fields
import pandas as pd
import msgpack
from StockDayData import StockDayData
import time

cache_url = Path(__file__).resolve().parent.parent / ".temp/data/cache"

def _dataclass_to_dict(obj):
    if isinstance(obj, StockDayData):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    return obj

def _dict_to_dataclass(obj):
    if isinstance(obj, dict):
        if "date" in obj and "code" in obj and "name" in obj:
            return StockDayData(**obj)
        return {k: _dict_to_dataclass(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dict_to_dataclass(item) for item in obj]
    return obj

def get_with_cache(file_name, func):
    start_time = datetime.now()
    try:
        with open(cache_url/file_name, "rb") as f:
            data = f.read()
            data = msgpack.unpackb(data, raw=False)
            def restore_dataframes(obj):
                if isinstance(obj, dict):
                    return {k: restore_dataframes(v) for k, v in obj.items()}
                elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                    cols = list(obj[0].keys()) if obj else []
                    if cols:
                        return pd.DataFrame(obj, columns=cols)
                return obj
            
            return _dict_to_dataclass(restore_dataframes(data))
    except Exception as e:
        print(f"读缓存{file_name}失败 原因：{e}")
    
    rs = func()
    packed = msgpack.packb(rs, default=_dataclass_to_dict, use_bin_type=True)
    with open(cache_url/file_name, "wb") as f:
        f.write(packed)
    print(f"写入缓存 {file_name} 耗时：{datetime.now()-start_time}ms")
    return rs

def get_with_cache_feather(file_name, func,sleep_time=0):
    start_time = datetime.now()
    try:
        df = pd.read_feather(cache_url/file_name)
        print(f"从缓存 {file_name} 读取数据耗时：{datetime.now()-start_time}ms")
        return df
    except Exception as e:
        print(f"读缓存{file_name}失败 原因：{e}")
    
    rs = func()
    rs.to_feather(cache_url/file_name)
    print(f"写入缓存 {file_name} 耗时：{datetime.now()-start_time}ms")
    if sleep_time>0:
        time.sleep(sleep_time)
    return rs
