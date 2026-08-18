# Contributing

Template improvements are welcome. If you started from
[**Use this template**](https://github.com/raphaelschwinger/torchrl-hydra-template/generate),
your repository has no git link to the upstream template by default — add a remote
manually (see below).

Not every change in a derived project belongs upstream. Contribute back when the
change is **template-worthy**: a generic algorithm, trainer or callback fix,
reusable environment config, documentation, or smoke test. Keep project-specific
work (pretraining experiments, paper configs, custom paths) in your own repo.

## Pulling template updates into your repo

One-time setup:

```shell
git remote add upstream https://github.com/raphaelschwinger/torchrl-hydra-template.git
git fetch upstream
```

How you sync depends on how you created your repo.

### Fork or clone of the template

Histories are already linked — merge or rebase works out of the box:

```shell
git checkout main
git merge upstream/main          # or: git rebase upstream/main
# resolve conflicts in shared files (src/, configs/, tests/)
pytest tests/test_smoke.py -v
```

### Created via [Use this template](https://github.com/raphaelschwinger/torchrl-hydra-template/generate)

GitHub starts a fresh repository with a new initial commit. The files match the
template, but git sees **no shared history**, so `git merge upstream/main` fails
with *refusing to merge unrelated histories*.

**Recommended — one-time history reconnect.** Rebase your project-specific
commits onto `upstream/main` so regular merges work from then on:

```shell
git remote add upstream https://github.com/raphaelschwinger/torchrl-hydra-template.git
git fetch upstream

# <initial-commit> = your repo's first commit (see git log --oneline --reverse)
git rebase --onto upstream/main <initial-commit> main
git push --force-with-lease origin main
```

Example: if `git log --oneline --reverse | head -1` shows `3a43db2 Initial
commit`, run `git rebase --onto upstream/main 3a43db2 main`.

After reconnecting, sync the same way as a fork:

```shell
git checkout main
git merge upstream/main
pytest tests/test_smoke.py -v
```

This rewrites history on `main`. Only run it once, early in the project, or
coordinate with collaborators before force-pushing.

**Without reconnecting** — pull in upstream changes selectively:

```shell
git cherry-pick <commit-sha>            # one upstream commit at a time

# — or — copy changed files manually
git diff upstream/main -- src/algorithms/dqn/dqn.py
pytest tests/test_smoke.py -v
```

Once you diverge, conflicts are likely — resolve them only in shared template
files.

## Feeding improvements back to the template

No separate fork clone is required. What matters is a **clean branch**: one
branched from `upstream/main` that contains only template-relevant changes, not
your full research history. You can create that branch in your existing derived
repo using the same `upstream` remote as above.

| Option | When to use |
|--------|-------------|
| **Pull request** | You have a focused, template-ready change |
| **GitHub issue** | Idea, bug report, or discussion before coding |
| **Cherry-pick / extract** | The improvement is buried in mixed commits on `main` |

**Pull request workflow** (works in your existing repo):

```shell
# one-time (if not already done for sync)
git remote add upstream https://github.com/raphaelschwinger/torchrl-hydra-template.git
git fetch upstream

# branch from upstream, not from your diverged main
git checkout -b contribute/my-fix upstream/main

# bring in your change (pick one):
git cherry-pick <commit-sha>            # if the commit is already template-only
# — or — copy changed files manually and commit

pytest tests/test_smoke.py -v
git push -u origin contribute/my-fix
# open PR: your-repo/contribute/my-fix → torchrl-hydra-template/main
```

Where to push and open the PR:

- **Maintainers / collaborators with write access:** push the branch directly to
  `upstream` and open an in-repo PR — no GitHub fork needed.
- **Everyone else:** push the branch to your repo (`origin`) and open a
  **cross-repo PR** from `your-repo:contribute/my-fix` →
  `torchrl-hydra-template:main`. GitHub supports this without cloning a
  separate fork.
- **Optional fork:** only if you prefer a dedicated template checkout; functionally
  equivalent to the branch-from-upstream flow above.

"Clean" here means **isolated diffs**, not a second repository.

**PR checklist:**

- Smoke test passes (`pytest tests/test_smoke.py -v`).
- Update `README.md` and `AGENTS.md` if you add or rename algorithms or change
  conventions (see [Adding a new algorithm](../README.md#adding-a-new-algorithm)).
- Keep PRs scoped — one algorithm, one bug fix, or one trainer improvement is
  easier to review than a large research dump.

**Issue workflow:** open an issue on
[torchrl-hydra-template](https://github.com/raphaelschwinger/torchrl-hydra-template/issues),
link to a minimal repro or branch in your repo, and explain why the change
belongs in the template rather than staying project-specific.
