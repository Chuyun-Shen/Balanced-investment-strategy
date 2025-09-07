# -*- coding: utf-8 -*-
"""
配置文件初始化
"""
import os
import importlib.util
import sys

# 优先尝试加载自定义配置，如果不存在则加载默认配置
def load_positions():
    custom_path = os.path.join(os.path.dirname(__file__), 'custom_positions.py')
    default_path = os.path.join(os.path.dirname(__file__), 'default_positions.py')
    
    if os.path.exists(custom_path):
        spec = importlib.util.spec_from_file_location("custom_positions", custom_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        return custom_module.current_positions
    else:
        spec = importlib.util.spec_from_file_location("default_positions", default_path)
        default_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(default_module)
        return default_module.current_positions
