

class Solution:

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init

        for _ in range(iterations):
            derivative = 2 * minimizer
            minimizer = minimizer - learning_rate * derivative

        return round(minimizer, 5)


if __name__ == "__main__":
    sol = Solution()
    res = sol.get_minimizer(0, 0.01, 5)
    print(res)

    sol = Solution()
    res = sol.get_minimizer(10, 0.01, 5)
    print(res)
