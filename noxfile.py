import os
import nox

nox.options.sessions = ["test_matrix"]

test_matrix_args = ['python', list('3.9 3.10 3.11 3.12 3.13'.split())]

@nox.session(venv_backend='uv')
@nox.parametrize(*test_matrix_args)
def test_matrix(session, python):
    """Run tests with different Python versions & extras."""
    # Allow filtering by passing arguments like: nox -s test_matrix -- 3.11 all
    if session.posargs and (str(python) != session.posargs[0]):
        session.skip(f"Skipping {python} because it's not in posargs {session.posargs}")
    session.install('.[dev]')
    args = ['pytest', f'-n{min(8, os.cpu_count() or 1)}', '--doctest-modules']
    session.run(*args)
    session.run('pip', 'wheel', '.')
