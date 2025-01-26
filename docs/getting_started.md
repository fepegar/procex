# Getting started

## Installation

For now, you can install `procex` from GitHub.
Soon, it will be available on PyPI.

=== "uv"

    ```shell
    uv add git+https://github.com/fepegar/procex
    ```

    !!! note
        You can install [uv](https://docs.astral.sh/uv/) with:

        ```shell
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

=== "pip"

    ```shell
    pip install git+https://github.com/fepegar/procex
    ```

## Usage

To get help, you can run `procex --help`.

=== "uv"

    ```shell
    uv run procex --help
    ```

    !!! note
        To quickly try `procex` without installing it, use [`uvx`](https://docs.astral.sh/uv/guides/tools/):

        ```shell
        uvx --from git+https://github.com/fepegar/procex procex --help
        ```

=== "pip"

    ```shell
    procex --help
    ```
