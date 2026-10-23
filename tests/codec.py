"""Data-only numeric wire encoding; array shape survives empty leading axes."""
import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise ValueError('Only numeric arrays are supported')
        return {'__wire_array__': {'shape': list(value.shape), 'data': encode(value.ravel().tolist())}}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, complex):
        return {'__wire_complex__': [value.real, value.imag]}
    if isinstance(value, tuple):
        return {'__wire_tuple__': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {'__wire_complex__'}:
            return complex(*value['__wire_complex__'])
        if set(value) == {'__wire_tuple__'}:
            return tuple(decode(x) for x in value['__wire_tuple__'])
        if set(value) == {'__wire_array__'}:
            data=value['__wire_array__']
            return np.asarray(decode(data['data'])).reshape(tuple(data['shape']))
        return {key: decode(x) for key,x in value.items()}
    return value
