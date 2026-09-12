# Contributing to Genesis

Thank you for your interest in contributing to Genesis! We welcome contributions from everyone. Please take a moment to review this guide to ensure a smooth collaboration.

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Submitting Code Changes](#submitting-code-changes)
- [Reviewing and Merging](#reviewing-and-merging)
- [Questions and Discussions](#questions-and-discussions)

---

## Reporting Bugs

- Before reporting a bug, please search through existing issues to check if it has already been reported.

- If the issue hasn't been reported yet, please use our issue templates to provide as much detail as possible in your report.

  ```markdown
  **Description**
  A clear and concise description of what the bug is.

  **To Reproduce**
  Example code or commands to reproduce the bug.

  **Expected behavior**
  A clear and concise description of what you expected to happen.

  **Screenshots**
  If applicable, add screenshots to help explain your problem.

  **Environment:**
   - OS: [e.g., Linux, macOS]
   - GPU/CPU: [e.g., A100, RTX 4090, M3pro]

  **Additional context**
  Add any other context about the problem here.
  ```

## Suggesting Features

- If you have a feature idea, please create an issue labeled `enhancement`.
- In the created issue, please provide context, expected outcomes, and potential.

## Submitting Code Changes

- We use the `pre-commit` configuration to automatically clean up code before committing. Install and run `pre-commit` as follows:
  1. Install `pre-commit`:

     ```bash
     pip install pre-commit
     ```

  2. Install hooks from the configuration file:

     ```bash
     pre-commit install
     ```

     After this, `pre-commit` will automatically check and clean up code whenever you make a commit.
- (Optional) You can run CI tests locally to ensure you pass the online CI checks.

  ```python
  pytest -v --forked -m required ./tests
  ```

- In the title of your Pull Request, please include [BUG FIX], [FEATURE], [MISC], [CHANGING] (the simulated physics change by default: a different model, or different default parameters) or [BREAKING] (API break) to indicate the purpose. The title is a sentence and ends with a period.
- Commit titles are plain single-line sentences without the bracket tag, also ending with a period.
- In the description, please provide example code or commands for testing.

## Reviewing and Merging

- PRs require at least one approval before merging.
- Automated checks (e.g., CI tests) must pass.
- Use `Squash and Merge` for a clean commit history.

## Questions and Discussions

- Use [Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions) for open-ended topics.
<!-- 
### Join Us
- Follow the project’s progress and updates on [channel/community link]. -->

---

We appreciate your contributions and look forward to collaborating with you!

Thank you,  
Genesis Maintainers
