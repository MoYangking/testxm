import inspect
from typing import List, Dict, Any, Type

class ParameterValidationMixin:
    def validate_and_convert_params(self, params: List[Any], param_type: Dict[str, Type]) -> Dict[str, Any]:
        '''将参数列表 params 根据 param_type 转换为参数字典。
        '''
        result = {}
        for i, (param_name, param_type_or_default_val) in enumerate(param_type.items()):
            if i >= len(params):
                if isinstance(param_type_or_default_val, Type) or param_type_or_default_val is inspect.Parameter.empty:
                    # 是类型
                    raise ValueError(f"参数 {param_name} 缺失")
                else:
                    # 是默认值
                    result[param_name] = param_type_or_default_val
            else:
                # 尝试强制转换
                try:
                    if param_type_or_default_val is None:
                        if params[i].isdigit():
                            result[param_name] = int(params[i])
                        else:
                            result[param_name] = params[i]
                    elif isinstance(param_type_or_default_val, str):
                        # 如果 param_type_or_default_val 是字符串，直接赋值
                        result[param_name] = params[i]
                    elif isinstance(param_type_or_default_val, int):
                        result[param_name] = int(params[i])
                    elif isinstance(param_type_or_default_val, float):
                        result[param_name] = float(params[i])
                    else:
                        result[param_name] = param_type_or_default_val(params[i])
                except ValueError:
                    raise ValueError(f"参数 {param_name} 类型错误")
        return result