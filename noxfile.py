import os
import nox

nox.options.sessions = ['build']
# nox.options.sessions = ['test_matrix', 'build']
sesh = dict(python=["3.9", "3.10", "3.11", "3.12", "3.13"] , venv_backend='uv')

@nox.session(**sesh)
def test_matrix(session):
    print(f'session {session.python}')
    nprocs = min(8, os.cpu_count() or 1)
    if session.posargs and (session.python) != session.posargs[0]:
        session.skip(f"Skipping {session.python} because it's not in posargs {session.posargs}")

    session.install('.[dev]')
    session.run(*f'pytest -n{nprocs} --doctest-modules --pyargs hgeom'.split())

@nox.session(**sesh)
def build(session):
    session.run(*'uv build .'.split())
