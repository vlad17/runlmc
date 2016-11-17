# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
This module contains a documentation-inheritance utility decorator,
:func:`inherit_doc`.
"""

import types

def inherit_doc(cls):
    """
    From StackOverflow.

    This will copy all the missing documentation for methods from the parent
    classes.

    :param type cls: class to fix up.
    :return type: the fixed class.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, property) and not func.fget.__doc__:
            for parent in cls.__bases__:
                parprop = getattr(parent, name, None)
                if parprop and getattr(parprop.fget, '__doc__', None):
                    newprop = property(fget=func.fget,
                                       fset=func.fset,
                                       fdel=func.fdel,
                                       doc=parprop.fget.__doc__)
                    setattr(cls, name, newprop)
                    break

    return cls
