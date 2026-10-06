"""GitHub App service (T3): receives webhooks, runs the graph already built in
T2 on the methods a pull request touches, and turns the graph's human-review
pause into an asynchronous one, resumed by a PR comment instead of a keypress.
"""
