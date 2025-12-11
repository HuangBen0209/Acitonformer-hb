import os

# 骨干网络（例如：卷积网络 / Transformer）
backbones = {}  # 存储所有注册的骨干网络类
def register_backbone(name):
    """
    注册骨干网络的装饰器。
    将一个类注册到 `backbones` 字典中，通过名称访问。
    """
    def decorator(cls):
        backbones[name] = cls  # 将类存储到字典中，键为名称
        return cls  # 返回原始类
    return decorator

# 颈部结构（例如：FPN）
necks = {}  # 存储所有注册的颈部结构类
def register_neck(name):
    """
    注册颈部结构的装饰器。
    将一个类注册到 `necks` 字典中，通过名称访问。
    """
    def decorator(cls):
        necks[name] = cls
        return cls
    return decorator

# 位置生成器（例如：点、线段等）
generators = {}  # 存储所有注册的位置生成器类
def register_generator(name):
    """
    注册位置生成器的装饰器。
    将一个类注册到 `generators` 字典中，通过名称访问。
    """
    def decorator(cls):
        generators[name] = cls
        return cls
    return decorator

# 元架构（每个模型的实际实现）
meta_archs = {}  # 存储所有注册的元架构类
def register_meta_arch(name):
    """
    注册元架构的装饰器。
    将一个类注册到 `meta_archs` 字典中，通过名称访问。
    """
    def decorator(cls):
        meta_archs[name] = cls
        return cls
    return decorator

# 构建函数
def make_backbone(name, **kwargs):
    """
    根据名称构建骨干网络。
    :param name: 骨干网络的名称
    :param kwargs: 传递给骨干网络类的参数
    :return: 实例化的骨干网络对象
    """
    backbone = backbones[name](**kwargs)  # 通过名称从字典中获取类并实例化
    return backbone

def make_neck(name, **kwargs):
    """
    根据名称构建颈部结构。
    :param name: 颈部结构的名称
    :param kwargs: 传递给颈部结构类的参数
    :return: 实例化的颈部结构对象
    """
    neck = necks[name](**kwargs)
    return neck

def make_meta_arch(name, **kwargs):
    """
    根据名称构建元架构。
    :param name: 元架构的名称
    :param kwargs: 传递给元架构类的参数
    :return: 实例化的元架构对象
    """
    meta_arch = meta_archs[name](**kwargs)
    return meta_arch

def make_generator(name, **kwargs):
    """
    根据名称构建位置生成器。
    :param name: 位置生成器的名称
    :param kwargs: 传递给位置生成器类的参数
    :return: 实例化的位置生成器对象
    """
    generator = generators[name](**kwargs)
    return generator