"""List the registered LOOT task IDs."""

import argparse

from mjlab.tasks.registry import list_tasks

import loot_bench  # noqa: F401  (registers all tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="List registered LOOT task IDs.")
    parser.add_argument(
        "--filter",
        default=None,
        help="Only print task IDs containing this substring "
        "(e.g. 'Domino' or 'Dk1').",
    )
    parser.add_argument(
        "--count", action="store_true", help="Only print the number of tasks."
    )
    args = parser.parse_args()

    tasks = [
        task_id
        for task_id in list_tasks()
        if args.filter is None or args.filter.lower() in task_id.lower()
    ]

    if args.count:
        print(f"Registered tasks: {str(len(tasks))}")
    else:
        print("Registered tasks:")
        for task_id in tasks:
            print(task_id)


if __name__ == "__main__":
    main()
