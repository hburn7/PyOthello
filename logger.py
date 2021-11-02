def log_comment(s: str, new_line: bool = True):
    """Logs a comment to the console in a referee-compatible format"""
    print(f'C {s}') if new_line else print(s, end='')