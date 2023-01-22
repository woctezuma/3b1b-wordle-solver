import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

from src.file import get_simulation_results_folder
from src.pattern import (
    get_pattern,
    get_possible_words,
    pattern_to_int_list,
    patterns_to_string,
)
from src.prior import get_frequency_based_priors, get_true_wordle_prior, get_word_list
from src.solver import brute_force_optimal_guess, optimal_guess

GAME_NAMES = ["wordle", "dungleon"]

# Run simulated wordle games


def simulate_games(
    game_name,
    first_guess=None,
    priors=None,
    look_two_ahead=False,
    optimize_for_uniform_distribution=False,
    second_guess_map=None,
    exclude_seen_words=False,
    test_set=None,
    shuffle=False,
    hard_mode=False,
    purely_maximize_information=False,
    brute_force_optimize=False,
    brute_force_depth=10,
    results_file=None,
    next_guess_map_file=None,
    quiet=False,
):
    all_words = get_word_list(game_name, short=False)
    short_word_list = get_word_list(game_name, short=True)

    if first_guess is None:
        first_guess = optimal_guess(
            all_words,
            all_words,
            priors,
            game_name=game_name,
            look_two_ahead=look_two_ahead,
            purely_maximize_information=purely_maximize_information,
            optimize_for_uniform_distribution=optimize_for_uniform_distribution,
        )

    if priors is None:
        priors = get_frequency_based_priors(game_name)

    if test_set is None or test_set[0] is None:
        test_set = short_word_list

    if shuffle:
        random.shuffle(test_set)

    seen = set()

    # Function for choosing the next guess, with a dict to cache
    # and reuse results that are seen multiple times in the sim
    next_guess_map = {}

    def get_next_guess(guesses, patterns, possibilities):
        phash = "".join(
            str(g) + "".join(map(str, pattern_to_int_list(p)))
            for g, p in zip(guesses, patterns, strict=True)
        )
        if second_guess_map is not None and len(patterns) == 1:
            next_guess_map[phash] = second_guess_map[patterns[0]]
        if phash not in next_guess_map:
            choices = all_words
            if hard_mode:
                for guess, pattern in zip(guesses, patterns, strict=True):
                    choices = get_possible_words(guess, pattern, choices, game_name)
            if brute_force_optimize:
                next_guess_map[phash] = brute_force_optimal_guess(
                    choices,
                    possibilities,
                    priors,
                    game_name=game_name,
                    n_top_picks=brute_force_depth,
                )
            else:
                next_guess_map[phash] = optimal_guess(
                    choices,
                    possibilities,
                    priors,
                    game_name,
                    look_two_ahead=look_two_ahead,
                    purely_maximize_information=purely_maximize_information,
                    optimize_for_uniform_distribution=optimize_for_uniform_distribution,
                )
        return next_guess_map[phash]

    # Go through each answer in the test set, play the game,
    # and keep track of the stats.
    scores = np.zeros(0, dtype=int)
    game_results = []
    score_dist = []
    total_guesses = 0
    for answer in tqdm(
        test_set,
        leave=False,
        desc=" Trying all wordle answers",
    ):
        guesses = []
        patterns = []
        possibility_counts = []
        possibilities = list(filter(lambda w: priors[w] > 0, all_words))

        if exclude_seen_words:
            possibilities = list(filter(lambda w: w not in seen, possibilities))

        score = 1
        guess = first_guess
        while guess != answer:
            pattern = get_pattern(guess, answer, game_name)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, possibilities, game_name)
            possibility_counts.append(len(possibilities))
            score += 1
            guess = get_next_guess(guesses, patterns, possibilities)

        # Accumulate stats
        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum()) for i in range(1, scores.max(initial=0) + 1)
        ]
        total_guesses = scores.sum()
        average = scores.mean()
        seen.add(answer)

        game_results.append(
            {
                "score": int(score),
                "answer": answer,
                "guesses": guesses,
                "patterns": list(map(int, patterns)),
                "reductions": possibility_counts,
            },
        )
        # Print outcome
        if not quiet:
            message = "\n".join(
                [
                    "",
                    f"Score: {score}",
                    f"Answer: {answer}",
                    f"Guesses: {guesses}",
                    f"Reductions: {possibility_counts}",
                    *patterns_to_string((*patterns, 3**5 - 1)).split("\n"),
                    *" " * (6 - len(patterns)),
                    f"Distribution: {score_dist}",
                    f"Total guesses: {total_guesses}",
                    f"Average: {average}",
                    *" " * 2,
                ],
            )
            if answer is not test_set[0]:
                # Move cursor back up to the top of the message
                n = len(message.split("\n")) + 1
                print("\033[F\033[K" * n)
            else:
                print("\r\033[K\n")
            print(message)

    final_result = {
        "score_distribution": score_dist,
        "total_guesses": int(total_guesses),
        "average_score": float(scores.mean()),
        "game_results": game_results,
    }

    # Save results
    for obj, file in (
        (final_result, results_file),
        (next_guess_map, next_guess_map_file),
    ):
        if file:
            path = os.path.join(get_simulation_results_folder(game_name), file)
            with open(path, "w", encoding="utf8") as fp:
                json.dump(obj, fp)

    return final_result, next_guess_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game-name",
        type=str,
        choices=GAME_NAMES,
        default="wordle",
        help="Game name",
    )
    parser.add_argument(
        "--first-guess",
        type=str,
        default=None,
        help="Pre-computed first guess",
    )
    parser.add_argument(
        "--test-answer",
        type=str,
        default=None,
        help="Solution with which to test the solver",
    )
    parser.add_argument(
        "--max-info",
        dest="purely_maximize_information",
        action="store_true",
        help="Purely maximize information",
    )
    parser.add_argument(
        "--flat-dist",
        dest="optimize_for_uniform_distribution",
        action="store_true",
        help="Optimize for uniform distribution",
    )
    parser.add_argument(
        "--look-ahead",
        dest="look_two_ahead",
        action="store_true",
        help="Look two ahead",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the test set",
    )
    parser.add_argument(
        "--brute-force",
        dest="brute_force_optimize",
        action="store_true",
        help="Perform brute-force optimization",
    )
    parser.add_argument(
        "--hard-mode",
        action="store_true",
        help="Play the hard mode",
    )
    args = parser.parse_args()

    results, decision_map = simulate_games(
        game_name=args.game_name,
        first_guess=args.first_guess,
        test_set=[args.test_answer],
        priors=get_true_wordle_prior(args.game_name),
        purely_maximize_information=args.purely_maximize_information,
        optimize_for_uniform_distribution=args.optimize_for_uniform_distribution,
        look_two_ahead=args.look_two_ahead,
        shuffle=args.shuffle,
        brute_force_optimize=args.brute_force_optimize,
        hard_mode=args.hard_mode,
    )
