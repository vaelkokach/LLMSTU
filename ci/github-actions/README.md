# GitHub Actions workflow (staged, not active)

`ci.yml` belongs at `.github/workflows/ci.yml`. It is staged here instead because
pushing a file to `.github/workflows/` requires a Personal Access Token carrying the
`workflow` scope, and the token used for this repository does not have it:

    ! [remote rejected] main -> main (refusing to allow a Personal Access Token to
      create or update workflow `.github/workflows/ci.yml` without `workflow` scope)

The workflow content is complete and unmodified. It is inert here — GitHub only runs
workflows under `.github/workflows/`.

## To activate

Preferred — grant the scope, then move the file back:

    # regenerate the PAT with the `workflow` scope ticked, then:
    mkdir -p .github/workflows && git mv ci/github-actions/ci.yml .github/workflows/ci.yml
    git commit -m "Activate CI workflow" && git push origin main

Alternative — add it through the GitHub web UI (Actions → new workflow) and paste
the contents of `ci.yml`. The web UI is not subject to the token scope check.

Nothing else in the repository depends on this file's location.
