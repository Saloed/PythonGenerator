import _pickle as P


def dump_object(obj, name):
    with open(name, 'wb') as f:
        P.dump(obj, f, protocol=2)


def load_object(name):
    with open(name, 'rb') as f:
        return P.load(f)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


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
