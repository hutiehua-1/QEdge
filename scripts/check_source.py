"""Validate Python syntax and local Markdown links."""
import ast
from pathlib import Path
import re
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
errors = []
sources = [p for p in root.rglob('*.py') if not any(x.startswith('.') for x in p.relative_to(root).parts)]
for path in sources:
    try:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except SyntaxError as exc:
        errors.append(str(exc))
for path in root.rglob('*.md'):
    if '.git' in path.parts or 'docs' in path.relative_to(root).parts:
        continue
    text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
    for target in re.findall(r'\]\(([^\s)]+)\)', text):
        if target.startswith(('http://','https://','mailto:','#')):
            continue
        local = unquote(target.split('#')[0])
        if local and not (path.parent/local).exists():
            errors.append(f'{path.relative_to(root)}: missing link {target}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(sources)} Python files parsed; local Markdown links resolved.')
