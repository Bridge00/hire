import errno
import os
import sys
import io
import signal
import functools
import threading
import ast
import astunparse


class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """
    This is a simple decorator that raises a TimeoutError if the function takes more than the specified time.
    Since some programs might end up not finishing, this is a good way to avoid infinite loops.
    :param seconds:
    :param error_message:
    :return:
    """
    def decorator(func):
        def _handle_timeout(signum=None, frame=None):
            if signum is not None:
                raise TimeoutError(error_message)
            # Windows: Interrupt main thread
            import _thread
            _thread.interrupt_main()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # --- UNIX: use SIGALRM ---
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)

            # --- WINDOWS: use threading.Timer ---
            timer = threading.Timer(seconds, _handle_timeout)
            timer.start()
            try:
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    # Convert interrupt to TimeoutError
                    raise TimeoutError(error_message)
            finally:
                timer.cancel()


        return wrapper

    return decorator

@timeout(10)
def timeout_exec(code):
    exec(code, globals(), globals())

@timeout(10)
def timeout_exec_with_return(code):
    loc = {}
    exec(code, loc, loc)
    return loc["my_new_var"]

def extract_test_code(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output_of_test(code, test):
    test_code = extract_test_code(test)
    failing_test = f'from typing import *\n\n{code}\n\nmy_new_var = {test_code}'
    output = timeout_exec_with_return(failing_test)
    return output

@timeout(10)
def run_code_with_io(code, input_str):
    """
    Executes code with redirected stdin and captures stdout.
    """
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    
    sys.stdin = io.StringIO(input_str)
    sys.stdout = io.StringIO()
    
    try:
        # Use a fresh dictionary for globals to avoid pollution, 
        # but include typical imports if needed.
        exec_globals = {"__builtins__": __builtins__}
        exec(code, exec_globals)
        output = sys.stdout.getvalue()
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        
    return output
