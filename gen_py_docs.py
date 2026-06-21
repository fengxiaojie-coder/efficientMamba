#!/usr/bin/env python3
"""Generate documentation for all Python files in the project."""

from pathlib import Path
import ast
import textwrap

repo = Path(r"e:/UclHomework/research project/efficientMamba")
files = [
    Path('main.py'),
    Path('src/eval.py'),
    Path('src/train.py'),
    Path('src/utils/visualize_roi_case.py'),
    Path('src/utils/make_subject_splits.py'),
    Path('src/datasets/ubfc_dataset.py'),
    Path('src/models/efficientphys_mamba.py'),
    Path('src/models/efficientphys_front.py'),
    Path('src/models/__init__.py'),
    Path('src/preprocessing/ubfc_to_tensors.py'),
    Path('src/preprocess/ubfc_rPPG_dataset_info.py'),
    Path('src/examples/run_frontend_example.py'),
    Path('src/preprocessing/roi_utils.py'),
    Path('src/preprocess/UBFC-rPPG_visualize.py'),
    Path('src/preprocessing/roi_extract.py'),
]

out_root = repo / 'docs' / 'python_files'
out_root.mkdir(parents=True, exist_ok=True)


def to_sig(node):
    """Convert a function def to a signature string."""
    parts = []
    args = node.args
    posonly = [a.arg for a in args.posonlyargs]
    regular = [a.arg for a in args.args]
    defaults = list(args.defaults)
    default_start = len(regular) - len(defaults)

    if posonly:
        for i, name in enumerate(posonly):
            parts.append(name)
        parts.append('/')

    for i, name in enumerate(regular):
        idx = i - default_start
        if idx >= 0:
            d = defaults[idx]
            try:
                dsrc = ast.unparse(d)
            except Exception:
                dsrc = '...'
            parts.append(f"{name}={dsrc}")
        else:
            parts.append(name)

    if args.vararg:
        parts.append('*' + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append('*')

    for kw, d in zip(args.kwonlyargs, args.kw_defaults):
        if d is None:
            parts.append(kw.arg)
        else:
            try:
                dsrc = ast.unparse(d)
            except Exception:
                dsrc = '...'
            parts.append(f"{kw.arg}={dsrc}")

    if args.kwarg:
        parts.append('**' + args.kwarg.arg)

    return f"{node.name}({', '.join(parts)})"


def first_line(doc):
    """Extract the first line of a docstring."""
    if not doc:
        return 'No docstring provided.'
    return doc.strip().splitlines()[0].strip()


def classify(rel):
    """Classify a file by its path."""
    s = str(rel).replace('\\', '/')
    if s.startswith('src/preprocessing/'):
        return 'Data preprocessing'
    if s.startswith('src/preprocess/'):
        return 'Data preparation helpers'
    if s.startswith('src/models/'):
        return 'Model definition'
    if s.startswith('src/datasets/'):
        return 'Dataset loading'
    if s.startswith('src/utils/'):
        return 'Utility script'
    if s.startswith('src/examples/'):
        return 'Example runner'
    if s == 'src/train.py':
        return 'Training entry'
    if s == 'src/eval.py':
        return 'Evaluation entry'
    if s == 'main.py':
        return 'Project quick entry'
    return 'General module'


index_lines = [
    '# Python File Documentation Index',
    '',
    'This index links to generated per-file documentation under `docs/python_files/`.',
    ''
]

for rel in files:
    fpath = repo / rel
    if not fpath.exists():
        print(f"Skipping {rel} (not found)")
        continue

    source = fpath.read_text(encoding='utf-8')
    tree = ast.parse(source)

    module_doc = ast.get_docstring(tree) or 'No module docstring provided.'
    imports = []
    constants = []
    classes = []
    funcs = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = ', '.join(alias.name + (f" as {alias.asname}" if alias.asname else '') for alias in node.names)
            imports.append(f"import {names}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            names = ', '.join(alias.name + (f" as {alias.asname}" if alias.asname else '') for alias in node.names)
            imports.append(f"from {mod} import {names}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        v = ast.unparse(node.value)
                    except Exception:
                        v = '...'
                    constants.append((target.id, v))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                try:
                    v = ast.unparse(node.value) if node.value else '...'
                except Exception:
                    v = '...'
                constants.append((node.target.id, v))
        elif isinstance(node, ast.ClassDef):
            methods = []
            for c in node.body:
                if isinstance(c, ast.FunctionDef):
                    methods.append((to_sig(c), first_line(ast.get_docstring(c))))
            classes.append((node.name, first_line(ast.get_docstring(node)), methods))
        elif isinstance(node, ast.FunctionDef):
            funcs.append((to_sig(node), first_line(ast.get_docstring(node))))

    out_path = out_root / rel.with_suffix('.md')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# {rel.as_posix()}")
    lines.append('')
    lines.append('## Role')
    lines.append(classify(rel))
    lines.append('')
    lines.append('## Module Summary')
    lines.append(module_doc.strip())
    lines.append('')

    lines.append('## Imports')
    if imports:
        for imp in imports:
            lines.append(f"- `{imp}`")
    else:
        lines.append('- None')
    lines.append('')

    lines.append('## Constants')
    if constants:
        for k, v in constants:
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append('- No explicit module-level constants detected')
    lines.append('')

    lines.append('## Functions')
    if funcs:
        for sig, doc in funcs:
            lines.append(f"- `{sig}`")
            lines.append(f"  - {doc}")
    else:
        lines.append('- No top-level functions')
    lines.append('')

    lines.append('## Classes')
    if classes:
        for cname, cdoc, methods in classes:
            lines.append(f"### `{cname}`")
            lines.append(cdoc)
            lines.append('')
            lines.append('Methods:')
            if methods:
                for msig, mdoc in methods:
                    lines.append(f"- `{msig}`")
                    lines.append(f"  - {mdoc}")
            else:
                lines.append('- No explicit methods documented')
            lines.append('')
    else:
        lines.append('- No classes')
        lines.append('')

    lines.append('## Notes')
    lines.append('- This document was generated from static AST inspection and module docstrings.')
    lines.append('- For runtime behavior, also cross-check the call sites in training/evaluation scripts.')
    lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Generated {out_path}")

    index_lines.append(f"- [{rel.as_posix()}](python_files/{rel.with_suffix('.md').as_posix()})")

index_path = repo / 'docs' / 'PythonFilesIndex.md'
index_path.write_text('\n'.join(index_lines) + '\n', encoding='utf-8')
print(f"\n✓ Generated {len(files)} file docs")
print(f"✓ Index: {index_path}")
