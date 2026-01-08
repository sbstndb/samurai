# Samurai Python API Migration Guide

## Overview

This directory contains scripts to help migrate Samurai Python code from the old API to the new `sam.field.*` namespace API.

## Migration Script

### `migrate_to_field_namespace.py`

Automatically converts old Samurai field API to the new `sam.field.*` API.

### What Gets Converted

#### ScalarField Constructors
```python
# OLD API
u = sam.ScalarField1D("u", mesh, 0.0)
u = sam.ScalarField2D("u", mesh, 0.0)
u = sam.ScalarField3D("u", mesh, 0.0)

# NEW API
u = sam.field.scalar(mesh, "u", init=0.0)
```

#### VectorField Constructors
```python
# OLD API
v = sam.VectorField1D_2("v", mesh, 0.0)
v = sam.VectorField2D_2("v", mesh, 0.0)
v = sam.VectorField2D_3("v", mesh, 0.0)
v = sam.VectorField3D_2("v", mesh, 0.0)
v = sam.VectorField3D_3("v", mesh, 0.0)

# NEW API
v = sam.field.vector(mesh, "v", n_components=2, init=0.0)
v = sam.field.vector(mesh, "v", n_components=3, init=0.0)
```

#### Factory Functions
```python
# OLD API
u = sam.make_scalar_field(mesh, "u", 1.0)
v = sam.make_vector_field(mesh, "v", 2, 0.0)

# NEW API
u = sam.field.scalar(mesh, "u", init=1.0)
v = sam.field.vector(mesh, "v", n_components=2, init=0.0)
```

### Usage

#### Migrate a Single File
```bash
python migrate_to_field_namespace.py example.py
```

#### Dry Run (Preview Changes)
```bash
python migrate_to_field_namespace.py example.py --dry-run
```

#### Migrate Multiple Files
```bash
python migrate_to_field_namespace.py file1.py file2.py file3.py
```

#### Migrate Directory
```bash
python migrate_to_field_namespace.py --dir python/examples
```

#### Dry Run on Directory
```bash
python migrate_to_field_namespace.py --dir python/examples --dry-run
```

### Safety Features

1. **Automatic Backups**: Creates `.bak` files before modifying
2. **Idempotent**: Safe to run multiple times (won't change already-converted code)
3. **Comment-Aware**: Skips code in comments
4. **String-Aware**: Skips API calls in string literals
5. **Dry-Run Mode**: Preview changes without modifying files

### What Gets Preserved

- Already converted code using `sam.field.scalar()` or `sam.field.vector()`
- Code in comments (`# u = sam.ScalarField2D(...)`)
- Code in string literals (`code = "sam.ScalarField2D(...)"`)
- All other code formatting and structure

### Output

The script shows:
- Number of changes made
- Line-by-line change summary
- Unified diff of changes
- Backup file location

### Example Output

```
======================================================================
Samurai Python API Migration Tool
======================================================================

example.py:
  Changes made (3 occurrences):
    Line 42: sam.ScalarField2D("u", mesh, 0.0)
           -> sam.field.scalar(mesh, "u", init=0.0)
    Line 43: sam.ScalarField2D("v", mesh, 0.0)
           -> sam.field.scalar(mesh, "v", init=0.0)
    Line 50: sam.VectorField2D_2("vel", mesh, 0.0)
           -> sam.field.vector(mesh, "vel", n_components=2, init=0.0)

  Diff:
    --- a/example.py
    +++ b/example.py
    @@ -39,8 +39,8 @@
         mesh = sam.MRMesh2D(box, config)

    -    u = sam.ScalarField2D("u", mesh, 0.0)
    -    v = sam.ScalarField2D("v", mesh, 0.0)
    +    u = sam.field.scalar(mesh, "u", init=0.0)
    +    v = sam.field.scalar(mesh, "v", init=0.0)

    -    vel = sam.VectorField2D_2("vel", mesh, 0.0)
    +    vel = sam.field.vector(mesh, "vel", n_components=2, init=0.0)

  Backup created: example.py.bak
  File updated successfully

======================================================================
Migration complete: 1 file(s) processed
======================================================================
```

## Rollback

If you need to revert changes, backup files are created with `.bak` extension:

```bash
# Restore from backup
mv example.py.bak example.py
```

## Integration with Git

### Recommended Workflow

1. **Create a feature branch** for migration:
   ```bash
   git checkout -b migrate-to-field-api
   ```

2. **Run migration in dry-run mode first**:
   ```bash
   python migrate_to_field_namespace.py --dir python/examples --dry-run
   ```

3. **Review the changes** in dry-run output

4. **Commit your current state** before migration:
   ```bash
   git add -A
   git commit -m "Before API migration"
   ```

5. **Run the migration**:
   ```bash
   python migrate_to_field_namespace.py --dir python/examples
   ```

6. **Review changes**:
   ```bash
   git diff
   ```

7. **Test the migrated code**:
   ```bash
   pytest python/tests/
   python python/examples/linear_convection.py
   ```

8. **Commit the migration**:
   ```bash
   git add -A
   git commit -m "feat: migrate to sam.field.* API"
   ```

9. **Clean up backup files** (after successful testing):
   ```bash
   find python -name '*.py.bak' -delete
   ```

### Git Ignore for Backups

Add to `.gitignore` to prevent committing backup files:
```
# Python migration script backups
*.py.bak
```

## Troubleshooting

### Script Doesn't Find Files

- Ensure file paths are absolute or relative to current directory
- Use `--dir` option for directories instead of listing files manually

### Unexpected Conversions

The script is conservative and only converts exact patterns:
- Must have `sam.` prefix (e.g., `sam.ScalarField2D`)
- Must match exact signature (name, mesh, init)
- Comments and strings are automatically skipped

### Already Converted Code

The script is idempotent - running it multiple times on the same file is safe:
- Already-converted `sam.field.scalar()` calls are preserved
- No duplicate conversions occur

## Advanced Usage

### Custom File Selection

Use `find` to select specific files:
```bash
# Only migrate example files
find python/examples -name '*.py' -exec migrate_to_field_namespace.py {} \;

# Only migrate test files
find python/tests -name 'test_*.py' -exec migrate_to_field_namespace.py {} \+
```

### Integration with Pre-Commit

Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: migrate-samurai-api
        name: Migrate Samurai API
        entry: python/python/scripts/migrate_to_field_namespace.py
        language: system
        files: '^python/.*\.py$'
        pass_filenames: true
```

## See Also

- [FIELD_NAMESPACE_API.md](../FIELD_NAMESPACE_API.md) - New API documentation
- [REFACTORING_GUIDE.md](../REFACTORING_GUIDE.md) - C++ refactoring guide
- [CLAUDE.md](../../CLAUDE.md) - Project documentation
