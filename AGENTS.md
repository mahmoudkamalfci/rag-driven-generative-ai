# Agent Instructions and Best Practices

## Dependency Management (Python / `uv`)
When adding new dependencies or installing packages for this project, you **must** use the `uv add` command rather than `uv pip install`. 

- **Production Dependencies**: Use `uv add <package_name>`
- **Development/Testing Dependencies**: Use `uv add --dev <package_name>`

**Why?** 
`uv add` automatically updates the `pyproject.toml` and `uv.lock` files to track the project's dependencies, similar to `npm install`. Using `uv pip install` only installs the package in the local environment without tracking it, which leads to missing dependencies for other developers or deployment environments.
