from __future__ import annotations

import argparse
from pathlib import Path

from mission1_2 import DEFAULT_INPUT, PROJECT_ROOT, compute_membership, read_ned_table


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "task3_common_members.csv"


EXPORT_COLUMNS = {
    "Object Name": "Object Name",
    "RA": "RA",
    "DEC": "DEC",
    "Redshift": "Redshift z",
    "Velocity": "Heliocentric Velocity",
    "Magnitude and Filter": "Magnitude",
    "Associations": "Associations",
}


def export_common_members(input_path: Path, output_path: Path) -> int:
    table = read_ned_table(input_path)
    results = compute_membership(table)

    common_members = table.loc[results["is_overlap"], list(EXPORT_COLUMNS)].copy()
    common_members = common_members.rename(columns=EXPORT_COLUMNS)
    common_members = common_members.sort_values(["RA", "DEC", "Object Name"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    common_members.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(common_members)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export targets that pass both Task 1 and Task 2 selections."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to NED result.txt. Defaults to data/result.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path. Defaults to data/task3_common_members.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_common_members(args.input, args.output)
    print(f"Exported {count} common Task 1/Task 2 members to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
