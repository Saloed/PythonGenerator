import pickle


def dump_object(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def load_object(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class Magic:
    def __init__(self, *args):
        for arg in args:
            for field_name, field_value in arg.__dict__.items():
                setattr(self, field_name, field_value)


class Struct:
    def __init__(self, **entries):
        self.__keys = entries.keys()
        self.__dict__.update(entries)

    def to_dict(self):
        return {
            key: self.__dict__[key]
            for key in self.__keys
        }


def dict_to_object(*_dict):
    if not _dict:
        return None
    elif len(_dict) == 1:
        _dict = _dict[0]
    else:
        tmp = {}
        for d in _dict:
            tmp.update(d)
        _dict = tmp
    return Struct(**_dict)


def dict_plus(*_dicts):
    if not _dicts:
        return None
    elif len(_dicts) == 1:
        return _dicts[0]
    else:
        tmp = {}
        for d in _dicts:
            tmp.update(d)
        return tmp
