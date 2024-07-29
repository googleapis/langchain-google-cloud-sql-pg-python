# DEVELOPER.md

## Versioning

This library follows [Semantic Versioning](http://semver.org/).

## Processes

### Conventional Commit messages

This repository uses tool [Release Please](https://github.com/googleapis/release-please) to create GitHub and PyPi releases. It does so by parsing your
git history, looking for [Conventional Commit messages](https://www.conventionalcommits.org/),
and creating release PRs.

Learn more by reading [How should I write my commits?](https://github.com/googleapis/release-please?tab=readme-ov-file#how-should-i-write-my-commits)

## Testing

### Run tests locally

1. Set environment variables for `INSTANCE_ID`, `DATABASE_ID`, `REGION`, `DB_USER`, `DB_PASSWORD`

1. Run pytest to automatically run all tests:

    ```bash
    pytest
    ```

### CI Platform Setup

Cloud Build is used to run tests against Google Cloud resources in test project: langchain-cloud-sql-testing.
Each test has a corresponding Cloud Build trigger, see [all triggers][triggers].
These tests are registered as required tests in `.github/sync-repo-settings.yaml`.

#### Trigger Setup

Cloud Build triggers (for Python versions 3.8 to 3.11) were created with the following specs:

```YAML
name: pg-integration-test-pr-py38
description: Run integration tests on PR for Python 3.8
filename: integration.cloudbuild.yaml
github:
  name: langchain-google-cloud-sql-pg-python
  owner: googleapis
  pullRequest:
    branch: .*
    commentControl: COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY
ignoredFiles:
  - docs/**
  - .kokoro/**
  - .github/**
  - "*.md"
substitutions:
  _CLUSTER_ID: <ADD_VALUE>
  _DATABASE_ID: <ADD_VALUE>
  _INSTANCE_ID: <ADD_VALUE>
  _REGION: us-central1
  _VERSION: "3.8"
```

Use `gcloud builds triggers import --source=trigger.yaml` to create triggers via the command line

#### Project Setup

1. Create an Cloud SQL for PostgreSQL instance and database
1. Setup Cloud Build triggers (above)

#### Run tests with Cloud Build

* Run integration test:

    ```bash
    gcloud builds submit --config integration.cloudbuild.yaml --region us-central1 --substitutions=_INSTANCE_ID=$INSTANCE_ID,_DATABASE_ID=$DATABASE_ID,_REGION=$REGION
    ```

#### Trigger

To run Cloud Build tests on GitHub from external contributors, ie RenovateBot, comment: `/gcbrun`.

#### Code Coverage
Please make sure your code is fully tested. The Cloud Build integration tests are run with the `pytest-cov` code coverage plugin. They fail for PRs with a code coverage less than the threshold specified in `.coveragerc`.  If your file is inside the main module and should be ignored by code coverage check, add it to the `omit` section of `.coveragerc`.

Check for code coverage report in any Cloud Build integration test log. 
Here is a breakdown of the report:
- `Stmts`:  lines of executable code (statements).
- `Miss`: number of lines not covered by tests.
- `Branch`: branches of executable code (e.g an if-else clause may count as 1 statement but 2 branches; test for both conditions to have both branches covered).
- `BrPart`: number of branches not covered by tests.
- `Cover`: average coverage of files.
- `Missing`: lines that are not covered by tests.

## Documentation

### LangChain Integration Docs

Google hosts documentation on LangChain's site via the [Google Provider][provider] page and individual integration pages:
[Vector Stores][vs], [Document Loaders][loaders], and [Memory][memory].

Currently, manual PRs are made to the [Langchain GitHub repo](https://github.com/langchain-ai/langchain).

### API Reference

#### Build the documentation
API docs are templated in the `docs/` directory.

To test locally, run: `nox -s docs`

The nox session, `docs`, is used to create HTML to publish to googleapis.dev
The nox session, `docfx`, is used to create YAML to publish to CGC.

#### Publish the documentation

The kokoro docs pipeline runs when a new release is created. See `.kokoro/` for the release pipeline.


[provider]: https://python.langchain.com/docs/integrations/platforms/google
[vs]: https://python.langchain.com/docs/integrations/vectorstores
[memory]: https://python.langchain.com/docs/integrations/memory
[loaders]: https://python.langchain.com/docs/integrations/document_loaders
[triggers]: https://console.cloud.google.com/cloud-build/triggers?e=13802955&project=langchain-cloud-sql-testing
[vectorstore]: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/docs/vector_store.ipynb
[loader]: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/docs/document_loader.ipynb
[history]: https://github.com/googleapis/langchain-google-cloud-sql-pg-python/tree/main/docs/chat_message_history.ipynb