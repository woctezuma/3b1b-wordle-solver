# 3b1b's Wordle solver

This repository contains Python code to solve Wordle puzzles with information theory.

## Requirements

- Install the latest version of [Python 3.X][python-download-url].
- Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To compute the optimal first guess, choose the game with `--game-name`:

```bash
python simulations.py --game-name wordle
```

```bash
python simulations.py --game-name dungleon
```

## References

- 3Blue1Brown, [*Solving Wordle using information theory*][youtube-video], posted on Youtube on February 6, 2022,
- [`3b1b/videos`][youtube-supplementary-code]: supplementary code (in Python) accompanying the aforementioned video,
- [`woctezuma/dungleon-bot`][dungleon-bot]: the application of different solvers to [Dungleon][dungleon-rules],
- [`woctezuma/Wordle-Bot`][wordle-bot-python-fork]: an extremely slow solver, mentioning some results.

<!-- Definitions -->

[python-download-url]: <https://www.python.org/downloads/>

[youtube-video]: <https://www.youtube.com/watch?v=v68zYyaEmEA>
[youtube-supplementary-code]: <https://github.com/3b1b/videos/tree/master/_2022/wordle>
[dungleon-bot]: <https://github.com/woctezuma/dungleon-bot>
[dungleon-rules]: <https://github.com/woctezuma/dungleon/wiki/Rules>
[wordle-bot-python-fork]: <https://github.com/woctezuma/Wordle-Bot>
