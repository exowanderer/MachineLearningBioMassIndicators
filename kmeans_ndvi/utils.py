from icecream import ic
ic.configureOutput(includeContext=True)


def info_message(message, end='\n', *args, **kwargs):
    ic.configureOutput(prefix='INFO | ')
    # print(f'[INFO] {message}', end=end)
    ic(message)


def warning_message(message, end='\n', *args, **kwargs):
    ic.configureOutput(prefix='WARNING | ')
    # print(f'[WARNING] {message}', end=end)
    ic(message)


def debug_message(message, end='\n', *args, **kwargs):
    ic.configureOutput(prefix='DEBUG | ')
    # print(f'[DEBUG] {message}', end=end)
    ic(message)
