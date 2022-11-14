# 3b1b's Wordle solver

[![Code Quality][codacy-image]][codacy]

This repository contains Python code to solve Wordle puzzles with information theory.

## Requirements

- Install the latest version of [Python 3.X][python-download-url].
- Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Choose the game with `--game-name`:

```bash
python simulations.py --game-name wordle
```

```bash
python simulations.py --game-name dungleon
```

Alternatively, run [`wordle_solver.ipynb`][colab-notebook]
[![Open In Colab][colab-badge]][colab-notebook]

## References

- 3Blue1Brown, [*Solving Wordle using information theory*][youtube-video], posted on Youtube on February 6, 2022,
- [`3b1b/videos`][youtube-supplementary-code]: supplementary code (in Python) accompanying the aforementioned video,
- [`woctezuma/dungleon-bot`][dungleon-bot]: the application of different solvers to [Dungleon][dungleon-rules],
- [`woctezuma/Wordle-Bot`][wordle-bot-python-fork]: an extremely slow solver, mentioning some results.

<!-- Definitions -->

[codacy]: <https://www.codacy.com/gh/woctezuma/3b1b-wordle-solver/dashboard>
[codacy-image]: <https://app.codacy.com/project/badge/Grade/ff156cc6b4604ba1a7527448480a118a>

[python-download-url]: <https://www.python.org/downloads/>
[colab-notebook]: <https://colab.research.google.com/github/woctezuma/3b1b-wordle-solver/blob/colab/wordle_solver.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

[youtube-video]: <https://www.youtube.com/watch?v=v68zYyaEmEA>
[youtube-supplementary-code]: <https://github.com/3b1b/videos/tree/master/_2022/wordle>
[dungleon-bot]: <https://github.com/woctezuma/dungleon-bot>
[dungleon-rules]: <https://github.com/woctezuma/dungleon/wiki/Rules>
[wordle-bot-python-fork]: <https://github.com/woctezuma/Wordle-Bot>
