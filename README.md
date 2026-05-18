# AST1601 Project 3

This repository is organized so original source files from collaborators can be
kept for reference while the maintained implementation is rewritten and archived
in Python.

## Directory layout

```text
AST1601_project3/
├── data/                  # Input data files
├── docs/                  # Reports and project documents
├── figs/                  # Generated figures and charts
├── src/
│   ├── python/            # Maintained Python rewrites
│   └── original/          # Preserved original source code by language
│       └── matlab/
└── requirements.txt       # Python dependencies
```

## Current Python workflow

Run the Coma cluster membership analysis from the project root:

```powershell
python .\src\python\mission1_2.py
```

By default, the script reads:

```text
data/result.txt
```

and writes figures to:

```text
figs/
```

Custom paths can be supplied when needed:

```powershell
python .\src\python\mission1_2.py --input .\data\result.txt --output-dir .\figs
```

## Collaboration rules

- Put all data files in `data/`.
- Put generated charts and figures in `figs/`.
- Put maintained Python rewrites in `src/python/`.
- Preserve original non-Python source code in `src/original/<language>/`.
- Keep reports and written deliverables in `docs/`.
