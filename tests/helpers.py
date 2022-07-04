"""
helper methods for tests
"""


# can be done using pytest.raises(e)
# def assert_exception(e):
#     """
#     decorator for tests that are designed to fail
#     :param e: specific exception class or tuple of exceptions to except
#     """
#     def wrapper(func):
#         def wrapped_func():
#             try:
#                 func()
#                 assert False
#             except e:
#                 assert True
#         return wrapped_func
#     return wrapper
