from itertools import combinations, permutations
import unittest

from check import minimum_total_distance


class MinimumDistanceTest(unittest.TestCase):
    def test_exhaustive_small_lanes_against_all_assignments(self):
        for size in range(4, 9):
            for count in range(1, min(3, size // 2) + 1):
                for flags in combinations(range(size), count):
                    available = sorted(set(range(size)) - set(flags))
                    brute = min(sum(abs(f - c) for f, c in zip(flags, assignment))
                                for assignment in permutations(available, count))
                    self.assertEqual(minimum_total_distance(flags, available), brute)

    def test_greedy_tie_break_can_increase_total(self):
        # Sequential nearest: 2->1, 3->5, 4->0 gives 7; optimum is 5.
        self.assertEqual(minimum_total_distance([2, 3, 4], [0, 1, 5]), 5)


if __name__ == '__main__':
    unittest.main()
