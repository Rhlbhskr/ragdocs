"""Create a source-only folder for manual GitHub uploads, excluding local data."""
from pathlib import Path
import shutil
import argparse

ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / 'github-upload'
DIRECTORIES = ('.github', '.streamlit', 'streamlit_app', 'app', 'docs', 'eval', 'samples', 'scripts', 'tests', 'web')
FILES = (
    '.dockerignore', '.env.example', '.gitignore', 'compose.yaml', 'Dockerfile',
    'README.md', 'render.yaml', 'requirements.txt', 'requirements-models.txt',
    'requirements-dev.txt', 'start.ps1', 'vercel.json', '.vercelignore',
    '.python-version', '.env.vercel.example', 'pytest.ini', 'start-streamlit.ps1',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Update source files in the existing export without deleting other files.')
    args = parser.parse_args()
    if DESTINATION.exists() and not args.refresh:
        raise SystemExit('github-upload already exists. Rename that folder before creating a fresh export.')
    DESTINATION.mkdir(exist_ok=args.refresh)
    for directory in DIRECTORIES:
        shutil.copytree(ROOT / directory, DESTINATION / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', 'secrets.toml'),
                        dirs_exist_ok=args.refresh)
    for filename in FILES:
        shutil.copy2(ROOT / filename, DESTINATION / filename)
    files = [p for p in DESTINATION.rglob('*') if p.is_file()]
    forbidden = {'.env', 'data', '.venv', '__pycache__', '.git', 'secrets.toml'}
    assert not any(forbidden.intersection(p.relative_to(DESTINATION).parts) for p in files)
    print(f'Created {DESTINATION}')
    print(f'{len(files)} source files; {sum(p.stat().st_size for p in files) / 1024 / 1024:.2f} MB total')
    print('Upload the contents of this folder, not the containing github-upload folder.')


if __name__ == '__main__':
    main()
