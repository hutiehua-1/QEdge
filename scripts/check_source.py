"""Validate Python syntax, copied-source integrity, and local Markdown links."""
import ast
import hashlib
import json
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
manifest = json.loads((root/'docs/source-manifest.json').read_text())
for item in manifest['files']:
    actual = hashlib.sha256((root/item['path']).read_bytes()).hexdigest()
    if actual != item['sha256']:
        errors.append('Original source changed: '+item['path'])
for path in root.rglob('*.md'):
    if '.git' in path.parts:
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
print(f'PASS: {len(sources)} Python files parsed; {len(manifest["files"])} original hashes matched; local Markdown links resolved.')
