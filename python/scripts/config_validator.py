#!/usr/bin/env python3
"""
Samurai Configuration Validator and Comparator

Utilitaire pour valider et comparer les configurations de simulation Samurai.
Détecte les incohérences, les magic numbers, et suggère des presets appropriés.

Usage:
    python config_validator.py --help
    python config_validator.py --validate examples/advection_2d.py
    python config_validator.py --compare examples/advection_2d.py examples/linear_convection.py
    python config_validator.py --suggest advection weno5
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ajouter le build directory au path
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)


@dataclass
class ExtractedConfig:
    """Configuration extraite d'un fichier Python"""
    filename: str
    min_level: Optional[int] = None
    max_level: Optional[int] = None
    max_stencil_size: Optional[int] = None
    cfl: Optional[float] = None
    epsilon: Optional[float] = None
    regularity: Optional[float] = None
    bc_order: Optional[int] = None
    scheme: Optional[str] = None  # "upwind", "weno5", etc.
    time_scheme: Optional[str] = None  # "euler", "SSPRK3"

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        return {
            'min_level': self.min_level,
            'max_level': self.max_level,
            'max_stencil_size': self.max_stencil_size,
            'cfl': self.cfl,
            'epsilon': self.epsilon,
            'regularity': self.regularity,
            'bc_order': self.bc_order,
            'scheme': self.scheme,
            'time_scheme': self.time_scheme,
        }

    def __str__(self) -> str:
        """Représentation string"""
        lines = [
            f"Configuration extraite de: {self.filename}",
            "=" * 70,
        ]
        for key, value in self.to_dict().items():
            if value is not None:
                lines.append(f"  {key:20s}: {value}")
        return "\n".join(lines)


class ConfigExtractor:
    """Extrait les configurations de fichiers Python"""

    # Patterns regex pour extraire les valeurs
    PATTERNS = {
        'min_level': r'min_level\s*=\s*(\d+)',
        'max_level': r'max_level\s*=\s*(\d+)',
        'max_stencil_size': r'max_stencil_size\s*=\s*(\d+)',
        'cfl': r'cfl\s*=\s*([\d.]+)',
        'epsilon': r'epsilon\s*=\s*([\d.eE-]+)',
        'regularity': r'regularity\s*=\s*([\d.]+)',
        'bc_order': r'order\s*=\s*(\d+)',
    }

    # Patterns pour détecter les schémas
    SCHEME_PATTERNS = {
        'upwind': r'\.upwind\s*\(',
        'weno5': r'weno5|make_convection_weno5',
        'convection': r'make_convection',
    }

    # Patterns pour détecter les schémas temporels
    TIME_SCHEME_PATTERNS = {
        'SSPRK3': r'3\.0\s*/\s*4\.0|1\.0\s*/\s*3\.0|2\.0\s*/\s*3\.0',
        'euler': r'unp1\s*=\s*u\s*-\s*dt\s*\*',
    }

    @classmethod
    def extract_from_file(cls, filepath: str) -> ExtractedConfig:
        """Extrait la configuration d'un fichier Python"""
        with open(filepath) as f:
            content = f.read()

        config = ExtractedConfig(filename=filepath)

        # Extraire les valeurs numériques
        for key, pattern in cls.PATTERNS.items():
            match = re.search(pattern, content)
            if match:
                value_str = match.group(1)
                # Convertir en int ou float selon la clé
                if key in ['epsilon', 'cfl', 'regularity']:
                    value = float(value_str)
                else:
                    value = int(value_str)
                setattr(config, key, value)

        # Détecter le schéma spatial
        for scheme, pattern in cls.SCHEME_PATTERNS.items():
            if re.search(pattern, content):
                config.scheme = scheme
                break

        # Détecter le schéma temporel
        for scheme, pattern in cls.TIME_SCHEME_PATTERNS.items():
            if re.search(pattern, content):
                config.time_scheme = scheme
                break

        return config


class ConfigValidator:
    """Valide une configuration Samurai"""

    # Valeurs typiques par schéma
    TYPICAL_VALUES = {
        'upwind': {
            'max_stencil_size': 2,
            'cfl': 0.5,
            'bc_order': 1,
            'time_scheme': 'euler',
        },
        'weno5': {
            'max_stencil_size': 6,
            'cfl': 0.95,
            'bc_order': 3,
            'time_scheme': 'SSPRK3',
        },
    }

    # Plages admissibles
    RANGES = {
        'cfl': (0.01, 1.0),
        'epsilon': (1e-8, 0.1),
        'regularity': (0.0, 3.0),
        'min_level': (0, 10),
        'max_level': (0, 15),
        'max_stencil_size': (2, 6),
        'bc_order': (1, 5),
    }

    @classmethod
    def validate(cls, config: ExtractedConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Valide une configuration.

        Returns:
            (is_valid, errors, warnings): Tuple
        """
        errors = []
        warnings = []

        # 1. Vérifier les plages admissibles
        for key, (min_val, max_val) in cls.RANGES.items():
            value = getattr(config, key, None)
            if value is not None:
                if value < min_val or value > max_val:
                    errors.append(f"{key}={value} hors plage [{min_val}, {max_val}]")

        # 2. Vérifier la cohérence min/max level
        if config.min_level is not None and config.max_level is not None:
            if config.min_level > config.max_level:
                errors.append(f"min_level ({config.min_level}) > max_level ({config.max_level})")
            elif config.max_level - config.min_level > 8:
                warnings.append(f"Écart max_level-min_level = {config.max_level - config.min_level} > 8 (grand)")

        # 3. Vérifier la cohérence schéma vs paramètres
        if config.scheme:
            typical = cls.TYPICAL_VALUES.get(config.scheme, {})

            # Vérifier max_stencil_size
            if config.scheme == 'weno5' and config.max_stencil_size is not None:
                if config.max_stencil_size < 6:
                    errors.append(f"WENO5 nécessite max_stencil_size >= 6, trouvé {config.max_stencil_size}")

            # Vérifier CFL
            if 'cfl' in typical and config.cfl is not None:
                typical_cfl = typical['cfl']
                if abs(config.cfl - typical_cfl) > 0.2:
                    warnings.append(f"CFL={config.cfl} inhabituel pour {config.scheme} (typique: {typical_cfl})")

            # Vérifier BC order
            if 'bc_order' in typical and config.bc_order is not None:
                typical_order = typical['bc_order']
                if config.bc_order < typical_order:
                    warnings.append(f"bc_order={config.bc_order} < {typical_order} peut réduire l'ordre pour {config.scheme}")

        # 4. Vérifier epsilon vs max_level
        if config.epsilon is not None and config.max_level is not None:
            if config.max_level >= 10 and config.epsilon > 1e-3:
                warnings.append(f"epsilon={config.epsilon} laxiste pour max_level={config.max_level} (suggère: < 1e-3)")
            elif config.max_level <= 5 and config.epsilon < 1e-4:
                warnings.append(f"epsilon={config.epsilon} très petit pour max_level={config.max_level} (gaspillage)")

        # 5. Vérifier la cohérence schéma temporel
        if config.time_scheme == 'euler' and config.cfl is not None:
            if config.cfl > 0.6:
                warnings.append(f"Euler forward avec CFL={config.cfl} > 0.6 peut être instable")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings


class ConfigSuggester:
    """Suggère des configurations appropriées"""

    @staticmethod
    def suggest_preset(problem_type: str, scheme: Optional[str] = None,
                       accuracy: str = 'default') -> Dict:
        """
        Suggère un preset basé sur le type de problème.

        Args:
            problem_type: Type de problème ("advection", "burgers", "diffusion")
            scheme: Schéma numérique ("upwind", "weno5")
            accuracy: Niveau de précision ("coarse", "default", "fine")

        Returns:
            Dict avec les paramètres suggérés
        """
        # Déterminer le schéma si non spécifié
        if scheme is None:
            if problem_type == "advection":
                scheme = "upwind"  # Par défaut
            else:
                scheme = "weno5"

        # Configurations de base par schéma
        base_configs = {
            'upwind': {
                'max_stencil_size': 2,
                'cfl': 0.5,
                'bc_order': 1,
                'time_scheme': 'euler',
            },
            'weno5': {
                'max_stencil_size': 6,
                'cfl': 0.95,
                'bc_order': 3,
                'time_scheme': 'SSPRK3',
            },
        }

        # Ajustements par précision
        accuracy_modifiers = {
            'coarse': {
                'min_level': 3,
                'max_level': 6,
                'epsilon': 5e-4,
            },
            'default': {
                'min_level': 4,
                'max_level': 10,
                'epsilon': 2e-4,
            },
            'fine': {
                'min_level': 5,
                'max_level': 12,
                'epsilon': 1e-4,
            },
        }

        # Ajustements par type de problème
        problem_modifiers = {
            'advection': {},
            'burgers': {
                'max_level': 7,  # Burgers nécessite moins de niveaux
            },
            'diffusion': {
                'cfl': 0.95,  # Diffusion plus stable
            },
        }

        # Fusionner les configurations
        suggested = {}
        suggested.update(base_configs[scheme])
        suggested.update(accuracy_modifiers[accuracy])
        suggested.update(problem_modifiers.get(problem_type, {}))

        # Ajouter des valeurs par défaut
        suggested.setdefault('regularity', 1.0)
        suggested.setdefault('relative_detail', False)

        return suggested


class ConfigComparator:
    """Compare deux configurations"""

    @staticmethod
    def compare(config1: ExtractedConfig, config2: ExtractedConfig) -> List[str]:
        """
        Compare deux configurations et retourne les différences.

        Returns:
            List[str]: Liste des différences
        """
        differences = []

        # Comparer tous les champs
        for key in config1.to_dict().keys():
            val1 = getattr(config1, key)
            val2 = getattr(config2, key)

            if val1 != val2:
                if val1 is None or val2 is None:
                    differences.append(f"  {key:20s}: {val1} vs {val2}")
                else:
                    # Calculer la différence relative pour les float
                    if isinstance(val1, float) and isinstance(val2, float):
                        if val1 != 0:
                            rel_diff = abs((val2 - val1) / val1) * 100
                            differences.append(f"  {key:20s}: {val1:.2e} vs {val2:.2e} ({rel_diff:+.1f}%)")
                        else:
                            differences.append(f"  {key:20s}: {val1:.2e} vs {val2:.2e}")
                    else:
                        differences.append(f"  {key:20s}: {val1} vs {val2}")

        return differences


def print_validation_report(config: ExtractedConfig, is_valid: bool,
                            errors: List[str], warnings: List[str]):
    """Affiche un rapport de validation"""
    print("\n" + "=" * 70)
    print("RAPPORT DE VALIDATION")
    print("=" * 70)
    print(config)
    print("\n" + "-" * 70)

    if is_valid:
        print("✅ Configuration VALIDE")
    else:
        print("❌ Configuration INVALIDE")

    if errors:
        print("\nERREURS:")
        for error in errors:
            print(f"  ❌ {error}")

    if warnings:
        print("\nAVERTISSEMENTS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    if not errors and not warnings:
        print("\n✅ Aucun problème détecté!")

    print("=" * 70 + "\n")


def print_comparison_report(config1: ExtractedConfig, config2: ExtractedConfig,
                           differences: List[str]):
    """Affiche un rapport de comparaison"""
    print("\n" + "=" * 70)
    print("RAPPORT DE COMPARAISON")
    print("=" * 70)
    print(f"Fichier 1: {config1.filename}")
    print(f"Fichier 2: {config2.filename}")
    print("-" * 70)

    if differences:
        print("DIFFÉRENCES:")
        for diff in differences:
            print(diff)
    else:
        print("✅ Les configurations sont IDENTIQUES")

    print("=" * 70 + "\n")


def print_suggestion_report(problem_type: str, scheme: Optional[str],
                            accuracy: str, suggested: Dict):
    """Affiche un rapport de suggestion"""
    print("\n" + "=" * 70)
    print("SUGGESTION DE CONFIGURATION")
    print("=" * 70)
    print(f"Type de problème: {problem_type}")
    if scheme:
        print(f"Schéma: {scheme}")
    print(f"Précision: {accuracy}")
    print("-" * 70)
    print("Paramètres suggérés:")

    for key, value in sorted(suggested.items()):
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.2e}")
        else:
            print(f"  {key:20s}: {value}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validateur et comparateur de configurations Samurai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s --validate python/examples/advection_2d.py
  %(prog)s --compare python/examples/advection_2d.py python/examples/linear_convection.py
  %(prog)s --suggest advection weno5 default
  %(prog)s --scan python/examples/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')

    # Commande validate
    validate_parser = subparsers.add_parser('validate', help='Valider une configuration')
    validate_parser.add_argument('file', help='Fichier Python à valider')

    # Commande compare
    compare_parser = subparsers.add_parser('compare', help='Comparer deux configurations')
    compare_parser.add_argument('file1', help='Premier fichier')
    compare_parser.add_argument('file2', help='Deuxième fichier')

    # Commande suggest
    suggest_parser = subparsers.add_parser('suggest', help='Suggérer une configuration')
    suggest_parser.add_argument('problem_type', help='Type de problème (advection, burgers, diffusion)')
    suggest_parser.add_argument('scheme', nargs='?', help='Schéma (upwind, weno5)')
    suggest_parser.add_argument('accuracy', default='default',
                               help='Précision (coarse, default, fine) [défaut: default]')

    # Commande scan
    scan_parser = subparsers.add_parser('scan', help='Scanner un répertoire')
    scan_parser.add_argument('directory', help='Répertoire à scanner')

    args = parser.parse_args()

    if args.command == 'validate':
        # Valider un fichier
        if not os.path.exists(args.file):
            print(f"Erreur: fichier '{args.file}' non trouvé", file=sys.stderr)
            sys.exit(1)

        config = ConfigExtractor.extract_from_file(args.file)
        is_valid, errors, warnings = ConfigValidator.validate(config)
        print_validation_report(config, is_valid, errors, warnings)

        sys.exit(0 if is_valid else 1)

    elif args.command == 'compare':
        # Comparer deux fichiers
        for f in [args.file1, args.file2]:
            if not os.path.exists(f):
                print(f"Erreur: fichier '{f}' non trouvé", file=sys.stderr)
                sys.exit(1)

        config1 = ConfigExtractor.extract_from_file(args.file1)
        config2 = ConfigExtractor.extract_from_file(args.file2)
        differences = ConfigComparator.compare(config1, config2)
        print_comparison_report(config1, config2, differences)

    elif args.command == 'suggest':
        # Suggérer une configuration
        suggested = ConfigSuggester.suggest_preset(
            args.problem_type, args.scheme, args.accuracy
        )
        print_suggestion_report(args.problem_type, args.scheme, args.accuracy, suggested)

    elif args.command == 'scan':
        # Scanner un répertoire
        if not os.path.isdir(args.directory):
            print(f"Erreur: '{args.directory}' n'est pas un répertoire", file=sys.stderr)
            sys.exit(1)

        # Trouver tous les fichiers .py
        py_files = list(Path(args.directory).rglob("*.py"))

        print(f"\nScan de {len(py_files)} fichiers Python dans {args.directory}")
        print("=" * 70)

        all_valid = True
        for py_file in py_files:
            config = ConfigExtractor.extract_from_file(str(py_file))
            is_valid, errors, warnings = ConfigValidator.validate(config)

            if not is_valid or warnings:
                print(f"\n{py_file}:")
                if not is_valid:
                    print("  ❌ INVALIDE")
                    for error in errors:
                        print(f"    {error}")
                    all_valid = False
                else:
                    print(f"  ⚠️  {len(warnings)} avertissement(s)")
        print("\n" + "=" * 70)
        if all_valid:
            print("✅ Tous les fichiers sont valides!")
        else:
            print("❌ Certains fichiers ont des erreurs")

        sys.exit(0 if all_valid else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
