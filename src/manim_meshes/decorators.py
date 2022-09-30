"""
A place for all decorators
"""
import warnings


def dangling_vert_decorator():
    """
    decorator for mesh to check for dangling vertices at the end of the decorated function
    check whether there are any dangling nodes - vertices that are not part of a face
    """
    def decorator_func(func):
        """the decorated function"""
        def wrapper_func(*args, **kwargs):
            mesh = args[0]
            return_value = func(*args, **kwargs)
            if mesh.test_for_dangling and mesh.dangling_vert_check():
                warnings.warn(f'Dangling vertices in {func.__name__}')
            return return_value
        return wrapper_func
    return decorator_func


def dangling_face_decorator():
    """
    decorator for mesh to check for dangling faces at the end of the decorated function
    check whether there are any dangling faces - faces that are not part of a part
    """
    def decorator_func(func):
        """the decorated function"""
        def wrapper_func(*args, **kwargs):
            mesh = args[0]
            return_value = func(*args, **kwargs)
            if mesh.test_for_dangling and mesh.dangling_face_check():
                warnings.warn(f'Dangling faces in {func.__name__}')
            return return_value
        return wrapper_func
    return decorator_func
