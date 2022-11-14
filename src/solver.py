import numpy as np
from tqdm import tqdm as ProgressDisplay

from src.entropy import (
    entropy_of_distributions,
    get_bucket_counts,
    get_entropies,
    get_pattern_distributions,
)
from src.pattern import get_pattern, get_possible_words, get_word_buckets
from src.prior import get_word_list

# Solvers


def get_weights(words, priors):
    frequencies = np.array([priors[word] for word in words])
    total = frequencies.sum()
    if total == 0:
        return np.zeros(frequencies.shape)
    return frequencies / total


def entropy_to_expected_score(ent):
    """
    Based on a regression associating entropies with typical scores
    from that point forward in simulated games, this function returns
    what the expected number of guesses required will be in a game where
    there's a given amount of entropy in the remaining possibilities.
    """
    # Assuming you can definitely get it in the next guess,
    # this is the expected score
    min_score = 2 ** (-ent) + 2 * (1 - 2 ** (-ent))

    # To account for the likely uncertainty after the next guess,
    # and knowing that entropy of 11.5 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 11.5)
    return min_score + 1.5 * ent / 11.5


def get_expected_scores(
    allowed_words,
    possible_words,
    priors,
    look_two_ahead=False,
    n_top_candidates_for_two_step=25,
):
    # Currenty entropy of distribution
    weights = get_weights(possible_words, priors)
    H0 = entropy_of_distributions(weights)
    H1s = get_entropies(allowed_words, possible_words, weights)

    word_to_weight = dict(zip(possible_words, weights))
    probs = np.array([word_to_weight.get(w, 0) for w in allowed_words])
    # If this guess is the true answer, score is 1. Otherwise, it's 1 plus
    # the expected number of guesses it will take after getting the corresponding
    # amount of information.
    expected_scores = probs + (1 - probs) * (1 + entropy_to_expected_score(H0 - H1s))

    if not look_two_ahead:
        return expected_scores

    # For the top candidates, refine the score by looking two steps out
    # This is currently quite slow, and could be optimized to be faster.
    # But why?
    sorted_indices = np.argsort(expected_scores)
    allowed_second_guesses = get_word_list()
    expected_scores += 1  # Push up the rest
    for i in ProgressDisplay(
        sorted_indices[:n_top_candidates_for_two_step],
        leave=False,
    ):
        guess = allowed_words[i]
        H1 = H1s[i]
        dist = get_pattern_distributions([guess], possible_words, weights)[0]
        buckets = get_word_buckets(guess, possible_words)
        second_guesses = [
            optimal_guess(allowed_second_guesses, bucket, priors, look_two_ahead=False)
            for bucket in buckets
        ]
        H2s = [
            get_entropies([guess2], bucket, get_weights(bucket, priors))[0]
            for guess2, bucket in zip(second_guesses, buckets)
        ]

        prob = word_to_weight.get(guess, 0)
        expected_scores[i] = sum(
            (
                # 1 times Probability guess1 is correct
                1 * prob,
                # 2 times probability guess2 is correct
                2
                * (1 - prob)
                * sum(
                    p * word_to_weight.get(g2, 0) for p, g2 in zip(dist, second_guesses)
                ),
                # 2 plus expected score two steps from now
                (1 - prob)
                * (
                    2
                    + sum(
                        p
                        * (1 - word_to_weight.get(g2, 0))
                        * entropy_to_expected_score(H0 - H1 - H2)
                        for p, g2, H2 in zip(dist, second_guesses, H2s)
                    )
                ),
            ),
        )
    return expected_scores


def get_score_lower_bounds(allowed_words, possible_words):
    """
    Assuming a uniform distribution on how likely each element
    of possible_words is, this gives the a lower boudn on the
    possible score for each word in allowed_words
    """
    bucket_counts = get_bucket_counts(allowed_words, possible_words)
    N = len(possible_words)
    # Probabilities of getting it in 1
    p1s = np.array([w in possible_words for w in allowed_words]) / N
    # Probabilities of getting it in 2
    p2s = bucket_counts / N - p1s
    # Otherwise, assume it's gotten in 3 (which is optimistics)
    p3s = 1 - bucket_counts / N
    return p1s + 2 * p2s + 3 * p3s


def optimal_guess(
    allowed_words,
    possible_words,
    priors,
    look_two_ahead=False,
    optimize_for_uniform_distribution=False,
    purely_maximize_information=False,
):
    if purely_maximize_information:
        if len(possible_words) == 1:
            return possible_words[0]
        weights = get_weights(possible_words, priors)
        ents = get_entropies(allowed_words, possible_words, weights)
        return allowed_words[np.argmax(ents)]

    # Just experimenting here...
    if optimize_for_uniform_distribution:
        expected_scores = get_score_lower_bounds(allowed_words, possible_words)
    else:
        expected_scores = get_expected_scores(
            allowed_words,
            possible_words,
            priors,
            look_two_ahead=look_two_ahead,
        )
    return allowed_words[np.argmin(expected_scores)]


def brute_force_optimal_guess(
    all_words,
    possible_words,
    priors,
    n_top_picks=10,
    display_progress=False,
):
    if len(possible_words) == 0:
        # Doesn't matter what to return in this case, so just default to first word in list.
        return all_words[0]
    # For the suggestions with the top expected scores, just
    # actually play the game out from this point to see what
    # their actual scores are, and minimize.
    expected_scores = get_score_lower_bounds(all_words, possible_words)
    top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]]
    true_average_scores = []
    if display_progress:
        iterable = ProgressDisplay(
            top_choices,
            desc=f"Possibilities: {len(possible_words)}",
            leave=False,
        )
    else:
        iterable = top_choices

    for next_guess in iterable:
        scores = []
        for answer in possible_words:
            score = 1
            possibilities = list(possible_words)
            guess = next_guess
            while guess != answer:
                possibilities = get_possible_words(
                    guess,
                    get_pattern(guess, answer),
                    possibilities,
                )
                # Make recursive? If so, we'd want to keep track of
                # the next_guess map and pass it down in the recursive
                # subcalls
                guess = optimal_guess(
                    all_words,
                    possibilities,
                    priors,
                    optimize_for_uniform_distribution=True,
                )
                score += 1
            scores.append(score)
        true_average_scores.append(np.mean(scores))
    return top_choices[np.argmin(true_average_scores)]
