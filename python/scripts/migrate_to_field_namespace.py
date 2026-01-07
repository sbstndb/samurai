#!/usr/bin/env python3
"""
Migration script to convert old Samurai Python API to new sam.field.* API.

This script automatically converts:
- ScalarField1D/2D/3D constructors -> sam.field.scalar()
- VectorField constructors -> sam.field.vector()
- make_scalar_field() -> sam.field.scalar()
- make_vector_field() -> sam.field.vector()

Usage:
    python migrate_to_field_namespace.py <file_path> [--dry-run]
    python migrate_to_field_namespace.py <file1> <file2> ... [--dry-run]
    python migrate_to_field_namespace.py --dir <directory> [--dry-run]

Options:
    --dry-run    Show changes without modifying files
    --dir        Process all Python files in a directory
"""

import sys
import os
import re
import argparse
import difflib
from pathlib import Path
from typing import List, Tuple, Optional


class FieldAPIMigrator:
    """Handles migration from old Samurai field API to new sam.field.* API."""

    # Conversion patterns: (pattern, replacement, description)
    CONVERSIONS = [
        # ScalarField direct constructors
        (
            r'\bsam\.\bScalarField1D\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.scalar(\2, "\1", init=\3)',
            'ScalarField1D(name, mesh, init) -> sam.field.scalar(mesh, name, init=...)'
        ),
        (
            r'\bsam\.\bScalarField2D\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.scalar(\2, "\1", init=\3)',
            'ScalarField2D(name, mesh, init) -> sam.field.scalar(mesh, name, init=...)'
        ),
        (
            r'\bsam\.\bScalarField3D\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.scalar(\2, "\1", init=\3)',
            'ScalarField3D(name, mesh, init) -> sam.field.scalar(mesh, name, init=...)'
        ),
        # ScalarField without init value (uses default)
        (
            r'\bsam\.\bScalarField1D\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.scalar(\2, "\1")',
            'ScalarField1D(name, mesh) -> sam.field.scalar(mesh, name)'
        ),
        (
            r'\bsam\.\bScalarField2D\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.scalar(\2, "\1")',
            'ScalarField2D(name, mesh) -> sam.field.scalar(mesh, name)'
        ),
        (
            r'\bsam\.\bScalarField3D\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.scalar(\2, "\1")',
            'ScalarField3D(name, mesh) -> sam.field.scalar(mesh, name)'
        ),

        # VectorField direct constructors (with component count in name)
        (
            r'\bsam\.\bVectorField1D_2\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\2, "\1", n_components=2, init=\3)',
            'VectorField1D_2(name, mesh, init) -> sam.field.vector(mesh, name, n_components=2, init=...)'
        ),
        (
            r'\bsam\.\bVectorField1D_3\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\2, "\1", n_components=3, init=\3)',
            'VectorField1D_3(name, mesh, init) -> sam.field.vector(mesh, name, n_components=3, init=...)'
        ),
        (
            r'\bsam\.\bVectorField2D_2\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\2, "\1", n_components=2, init=\3)',
            'VectorField2D_2(name, mesh, init) -> sam.field.vector(mesh, name, n_components=2, init=...)'
        ),
        (
            r'\bsam\.\bVectorField2D_3\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\2, "\1", n_components=3, init=\3)',
            'VectorField2D_3(name, mesh, init) -> sam.field.vector(mesh, name, n_components=3, init=...)'
        ),
        (
            r'\bsam\.\bVectorField3D_2\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\2, "\1", n_components=2, init=\3)',
            'VectorField3D_2(name, mesh, init) -> sam.field.vector(mesh, name, n_components=2, init=...)'
        ),
        (
            r'\bsam\.\bVectorField3D_3\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\2, "\1", n_components=3, init=\3)',
            'VectorField3D_3(name, mesh, init) -> sam.field.vector(mesh, name, n_components=3, init=...)'
        ),
        # VectorField without init value
        (
            r'\bsam\.\bVectorField1D_2\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.vector(\2, "\1", n_components=2)',
            'VectorField1D_2(name, mesh) -> sam.field.vector(mesh, name, n_components=2)'
        ),
        (
            r'\bsam\.\bVectorField1D_3\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.vector(\2, "\1", n_components=3)',
            'VectorField1D_3(name, mesh) -> sam.field.vector(mesh, name, n_components=3)'
        ),
        (
            r'\bsam\.\bVectorField2D_2\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.vector(\2, "\1", n_components=2)',
            'VectorField2D_2(name, mesh) -> sam.field.vector(mesh, name, n_components=2)'
        ),
        (
            r'\bsam\.\bVectorField2D_3\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.vector(\2, "\1", n_components=3)',
            'VectorField2D_3(name, mesh) -> sam.field.vector(mesh, name, n_components=3)'
        ),
        (
            r'\bsam\.\bVectorField3D_2\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.vector(\2, "\1", n_components=2)',
            'VectorField3D_2(name, mesh) -> sam.field.vector(mesh, name, n_components=2)'
        ),
        (
            r'\bsam\.\bVectorField3D_3\s*\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\)',
            r'sam.field.vector(\2, "\1", n_components=3)',
            'VectorField3D_3(name, mesh) -> sam.field.vector(mesh, name, n_components=3)'
        ),

        # Factory functions
        (
            r'\bsam\.\bmake_scalar_field\s*\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*,\s*([^)]+)\)',
            r'sam.field.scalar(\1, "\2", init=\3)',
            'make_scalar_field(mesh, name, init) -> sam.field.scalar(mesh, name, init=...)'
        ),
        (
            r'\bsam\.\bmake_scalar_field\s*\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*\)',
            r'sam.field.scalar(\1, "\2")',
            'make_scalar_field(mesh, name) -> sam.field.scalar(mesh, name)'
        ),
        (
            r'\bsam\.\bmake_vector_field\s*\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*,\s*(\d+)\s*,\s*([^)]+)\)',
            r'sam.field.vector(\1, "\2", n_components=\3, init=\4)',
            'make_vector_field(mesh, name, n_comp, init) -> sam.field.vector(mesh, name, n_components=..., init=...)'
        ),
        (
            r'\bsam\.\bmake_vector_field\s*\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*,\s*(\d+)\s*\)',
            r'sam.field.vector(\1, "\2", n_components=\3)',
            'make_vector_field(mesh, name, n_comp) -> sam.field.vector(mesh, name, n_components=...)'
        ),
    ]

    def __init__(self, content: str, filepath: str):
        """Initialize migrator with file content.

        Args:
            content: Original file content
            filepath: Path to file (for error reporting)
        """
        self.original_content = content
        self.content = content
        self.filepath = filepath
        self.changes: List[Tuple[int, str, str]] = []
        self.line_mapping: List[int] = []

    def _build_line_mapping(self):
        """Build mapping from character position to line number."""
        self.line_mapping = [1]  # Position 0 is line 1
        line_num = 1
        for i, char in enumerate(self.original_content):
            if char == '\n':
                line_num += 1
            self.line_mapping.append(line_num)

    def _get_line_number(self, pos: int) -> int:
        """Get line number for a character position."""
        if pos < len(self.line_mapping):
            return self.line_mapping[pos]
        return self.line_mapping[-1]

    def is_in_comment(self, pos: int) -> bool:
        """Check if position is within a comment.

        Args:
            pos: Character position in content

        Returns:
            True if position is in a comment
        """
        # Simple check: look backwards for # or string delimiters
        line_start = self.original_content.rfind('\n', 0, pos) + 1
        line_before = self.original_content[line_start:pos]

        # Check for # comment
        if '#' in line_before:
            return True

        return False

    def is_in_string(self, pos: int) -> bool:
        """Check if position is within a string literal.

        Args:
            pos: Character position in content

        Returns:
            True if position is in a string literal
        """
        # Count quotes before this position
        before = self.original_content[:pos]
        single_quotes = before.count("'") - before.count("\\'")
        double_quotes = before.count('"') - before.count('\\"')

        # If odd number of unescaped quotes, we're in a string
        return single_quotes % 2 == 1 or double_quotes % 2 == 1

    def should_skip_position(self, pos: int) -> bool:
        """Check if conversion should be skipped at this position.

        Args:
            pos: Character position

        Returns:
            True if skip (in comment or string)
        """
        return self.is_in_comment(pos) or self.is_in_string(pos)

    def apply_conversions(self) -> str:
        """Apply all conversion patterns to content.

        Returns:
            Modified content
        """
        self._build_line_mapping()

        for pattern, replacement, description in self.CONVERSIONS:
            matches = list(re.finditer(pattern, self.content))

            # Process matches in reverse order to preserve positions
            for match in reversed(matches):
                # Check if we should skip this match
                if self.should_skip_position(match.start()):
                    continue

                # Apply replacement
                old_text = match.group(0)
                new_text = re.sub(pattern, replacement, old_text)

                # Record change
                line_num = self._get_line_number(match.start())
                self.changes.append((line_num, old_text, new_text))

                # Apply replacement to content
                start, end = match.span()
                self.content = self.content[:start] + new_text + self.content[end:]

        return self.content

    def has_changes(self) -> bool:
        """Check if any changes were made."""
        return len(self.changes) > 0

    def get_diff(self) -> List[str]:
        """Generate unified diff of changes.

        Returns:
            List of diff lines
        """
        original_lines = self.original_content.splitlines(keepends=True)
        modified_lines = self.content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f'a/{self.filepath}',
            tofile=f'b/{self.filepath}',
            lineterm=''
        )

        return list(diff)

    def get_change_summary(self) -> List[str]:
        """Get summary of changes made.

        Returns:
            List of change descriptions
        """
        summary = []
        for line_num, old_text, new_text in sorted(self.changes):
            # Truncate long texts for readability
            old_display = old_text[:80] + '...' if len(old_text) > 80 else old_text
            summary.append(f"  Line {line_num}: {old_display}")
            summary.append(f"         -> {new_text}")

        return summary


def backup_file(filepath: Path) -> Path:
    """Create backup of file with .bak extension.

    Args:
        filepath: Path to file to backup

    Returns:
        Path to backup file
    """
    backup_path = filepath.with_suffix(filepath.suffix + '.bak')
    content = filepath.read_text()
    backup_path.write_text(content)
    return backup_path


def migrate_file(filepath: Path, dry_run: bool = False) -> bool:
    """Migrate a single file.

    Args:
        filepath: Path to file
        dry_run: If True, don't modify files

    Returns:
        True if changes were made/needed
    """
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return False

    if not filepath.is_file():
        print(f"Error: Not a file: {filepath}")
        return False

    # Read file
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return False

    # Apply migrations
    migrator = FieldAPIMigrator(content, str(filepath))
    modified_content = migrator.apply_conversions()

    if not migrator.has_changes():
        print(f"  No changes needed")
        return False

    # Show changes
    print(f"\n{filepath}:")
    print(f"  Changes made ({len(migrator.changes)} occurrences):")
    for change in migrator.get_change_summary():
        print(f"    {change}")

    # Show diff
    diff = migrator.get_diff()
    if diff:
        print(f"\n  Diff:")
        for line in diff:
            print(f"    {line}")

    # Write file if not dry run
    if not dry_run:
        # Create backup
        backup_path = backup_file(filepath)
        print(f"\n  Backup created: {backup_path}")

        # Write modified content
        filepath.write_text(modified_content)
        print(f"  File updated successfully")
    else:
        print(f"\n  [DRY RUN] File not modified")

    return True


def migrate_directory(directory: Path, dry_run: bool = False) -> int:
    """Migrate all Python files in a directory.

    Args:
        directory: Directory path
        dry_run: If True, don't modify files

    Returns:
        Number of files migrated
    """
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 0

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return 0

    # Find all Python files
    py_files = list(directory.rglob('*.py'))

    if not py_files:
        print(f"No Python files found in {directory}")
        return 0

    print(f"Found {len(py_files)} Python files in {directory}")
    print()

    count = 0
    for py_file in sorted(py_files):
        if migrate_file(py_file, dry_run):
            count += 1
        print()

    return count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate Samurai Python code to new sam.field.* API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single file
  python migrate_to_field_namespace.py example.py

  # Dry run (show changes without modifying)
  python migrate_to_field_namespace.py example.py --dry-run

  # Migrate multiple files
  python migrate_to_field_namespace.py file1.py file2.py file3.py

  # Migrate all files in a directory
  python migrate_to_field_namespace.py --dir python/examples

  # Dry run on directory
  python migrate_to_field_namespace.py --dir python/examples --dry-run

Conversions performed:
  - sam.ScalarField1D/2D/3D(name, mesh, init) -> sam.field.scalar(mesh, name, init=...)
  - sam.VectorField*_(name, mesh, init) -> sam.field.vector(mesh, name, n_components=..., init=...)
  - sam.make_scalar_field(mesh, name, init) -> sam.field.scalar(mesh, name, init=...)
  - sam.make_vector_field(mesh, name, n, init) -> sam.field.vector(mesh, name, n_components=..., init=...)
        """
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='Python files to migrate'
    )

    parser.add_argument(
        '--dir',
        dest='directory',
        help='Directory containing Python files to migrate'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show changes without modifying files'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.directory and args.files:
        print("Error: Cannot use both --dir and file arguments")
        return 1

    if not args.directory and not args.files:
        parser.print_help()
        return 1

    print("=" * 70)
    print("Samurai Python API Migration Tool")
    print("=" * 70)
    print()

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()

    # Process files
    if args.directory:
        directory = Path(args.directory)
        count = migrate_directory(directory, args.dry_run)
        print()
        print("=" * 70)
        print(f"Migration complete: {count} file(s) processed")
    else:
        count = 0
        for filepath_str in args.files:
            filepath = Path(filepath_str)
            if migrate_file(filepath, args.dry_run):
                count += 1

        print()
        print("=" * 70)
        print(f"Migration complete: {count} file(s) processed")

    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
