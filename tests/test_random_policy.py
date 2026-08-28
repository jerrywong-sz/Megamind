import random

from src.augmentations import _choose_transform_count


def test_random_transform_count_policy():
    random.seed(0)
    total = 10000
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for _ in range(total):
        counts[_choose_transform_count()] += 1

    percentages = {
        count: seen / total * 100
        for count, seen in counts.items()
    }

    print(f"zero transforms: {percentages[0]:.1f}%")
    print(f"one transform: {percentages[1]:.1f}%")
    print(f"two transforms: {percentages[2]:.1f}%")
    print(f"three transforms: {percentages[3]:.1f}%")

    assert 27 <= percentages[0] <= 33
    assert 47 <= percentages[1] <= 53
    assert 12 <= percentages[2] <= 18
    assert 3 <= percentages[3] <= 7
