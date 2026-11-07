#!/usr/bin/env bash
# Deploy a git ref to the Hugging Face Space without repo-only files (the model-card source).
#
#   scripts/deploy_space.sh            # deploy main
#   scripts/deploy_space.sh my-branch  # deploy another ref
#   DRY_RUN=1 scripts/deploy_space.sh  # list what would be left out, push nothing
#
# Files in EXCLUDE never reach the Space; it gets a commit
# with the same tree minus EXCLUDE, on top of the Space's own history. After the first deploy,
# a plain `git push space main` is rejected (non-fast-forward): always deploy with this script.
set -euo pipefail

REF="${1:-main}"
REMOTE="${SPACE_REMOTE:-space}"
# Repo-only files. Keep README.md (Space card) and examples/**/CREDITS.md (CC BY-SA credits).
EXCLUDE=(MODEL_CARD.md graphify-out)

git fetch -q "$REMOTE"
INDEX="$(mktemp -u)"
trap 'rm -f "$INDEX"' EXIT
export GIT_INDEX_FILE="$INDEX"
git read-tree "$REF"
REMOVED="$(git ls-files -- "${EXCLUDE[@]}")"
if [ -n "$REMOVED" ]; then
  printf '%s\n' "$REMOVED" | git update-index --force-remove --stdin  # temp index only
fi
TREE="$(git write-tree)"
PARENT="$(git rev-parse "$REMOTE/main")"

echo "Deploying $REF ($(git rev-parse --short "$REF")) to $REMOTE/main without:"
printf '  %s\n' ${REMOVED:-"(nothing to leave out)"}
if [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then
  echo "Space already has exactly this content; nothing to push."
  exit 0
fi
if [ -n "${DRY_RUN:-}" ]; then
  echo "(dry run: nothing pushed)"
  exit 0
fi
COMMIT="$(git commit-tree "$TREE" -p "$PARENT" -m "deploy: $(git rev-parse --short "$REF") $(git log -1 --format=%s "$REF")" -m "Left out: $(echo $REMOVED)")"
git push "$REMOTE" "$COMMIT:refs/heads/main"
