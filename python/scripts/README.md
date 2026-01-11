# Samurai Python Scripts

This directory contains utility scripts for working with Samurai Python bindings.

## Available Scripts

### `migrate_to_field_namespace.py`

**Purpose:** Automatically migrate Samurai Python code from old API to new `sam.field.*` namespace API.

**Quick Start:**
```bash
# Preview changes without modifying files
python migrate_to_field_namespace.py example.py --dry-run

# Migrate a single file
python migrate_to_field_namespace.py example.py

# Migrate all files in a directory
python migrate_to_field_namespace.py --dir python/examples
```

**What It Converts:**
- `sam.ScalarField1D/2D/3D(name, mesh, init)` → `sam.field.scalar(mesh, name, init=...)`
- `sam.VectorField*_(name, mesh, init)` → `sam.field.vector(mesh, name, n_components=..., init=...)`
- `sam.make_scalar_field(mesh, name, init)` → `sam.field.scalar(mesh, name, init=...)`
- `sam.make_vector_field(mesh, name, n, init)` → `sam.field.vector(mesh, name, n_components=..., init=...)`

**Safety Features:**
- Automatic `.bak` backup files created before modification
- Idempotent (safe to run multiple times)
- Skips code in comments and string literals
- Shows comprehensive diff output
- Dry-run mode for previewing changes

**Documentation:** See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed usage.

### `config_validator.py`

**Purpose:** Validate Samurai mesh configurations and check for common issues.

**Usage:**
```bash
python config_validator.py <config_file>
```

**Features:**
- Validates mesh configuration parameters
- Checks for inconsistent settings
- Reports potential issues
- Suggests corrections

## Quick Reference

### Migration Command Examples

```bash
# Single file, preview mode
migrate_to_field_namespace.py myfile.py --dry-run

# Multiple files
migrate_to_field_namespace.py file1.py file2.py file3.py

# Entire directory
migrate_to_field_namespace.py --dir python/examples

# Recursive directory with find
find python -name '*.py' -exec migrate_to_field_namespace.py {} \+
```

### Migration Workflow

1. **Review changes first:**
   ```bash
   python migrate_to_field_namespace.py --dir python/tests --dry-run
   ```

2. **Commit current state:**
   ```bash
   git add -A && git commit -m "Before API migration"
   ```

3. **Run migration:**
   ```bash
   python migrate_to_field_namespace.py --dir python/tests
   ```

4. **Review and test:**
   ```bash
   git diff
   pytest python/tests/
   ```

5. **Commit migration:**
   ```bash
   git add -A && git commit -m "feat: migrate to sam.field.* API"
   ```

6. **Clean up backups** (after successful testing):
   ```bash
   find python -name '*.py.bak' -delete
   ```

## Troubleshooting

### Migration Script Issues

**Problem:** Script doesn't find my files
- **Solution:** Use absolute paths or `--dir` option

**Problem:** Too much output
- **Solution:** Use `2>&1 | grep -E "(changes made|Migration complete)"` to filter

**Problem:** Need to revert
- **Solution:** Restore from `.bak` files: `mv file.py.bak file.py`

### Config Validator Issues

**Problem:** Validator reports errors
- **Solution:** Review the error messages and adjust your configuration

**Problem:** False positives
- **Solution:** Some warnings may be informational - review your specific use case

## Contributing

When adding new scripts:

1. Include comprehensive help text (`--help`)
2. Add usage examples in this README
3. Test with dry-run or preview modes when applicable
4. Follow Python best practices (PEP 8, type hints)
5. Add docstrings to all functions and classes

## Related Documentation

- [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - Detailed migration guide
- [FIELD_NAMESPACE_API.md](../FIELD_NAMESPACE_API.md) - New API documentation
- [REFACTORING_GUIDE.md](../REFACTORING_GUIDE.md) - C++ refactoring guide
- [CLAUDE.md](../../CLAUDE.md) - Project documentation

## Support

For issues or questions:
1. Check the script's `--help` output
2. Review relevant documentation
3. Open an issue on GitHub: https://github.com/hpc-maths/samurai/issues
