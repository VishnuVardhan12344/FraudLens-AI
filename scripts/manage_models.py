
import argparse
import json

from fraudlens.model_registry import (
    register_model_bundle,
    promote_candidate,
    rollback_model,
    deploy_champion,
    registry_status,
)


def main():

    parser = argparse.ArgumentParser(
        description="FraudLens Model Management"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    register = commands.add_parser(
        "register"
    )

    register.add_argument(
        "--path",
        required=True,
    )

    register.add_argument(
        "--baseline",
        action="store_true",
    )

    promote = commands.add_parser(
        "promote"
    )

    promote.add_argument(
        "--version",
        required=True,
    )

    promote.add_argument(
        "--approve",
        action="store_true",
    )

    commands.add_parser("rollback")
    commands.add_parser("deploy")
    commands.add_parser("status")

    args = parser.parse_args()

    if args.command == "register":

        result = register_model_bundle(
            args.path,
            baseline=args.baseline,
        )

    elif args.command == "promote":

        result = promote_candidate(
            args.version,
            approved=args.approve,
        )

    elif args.command == "rollback":

        result = rollback_model()

    elif args.command == "deploy":

        result = deploy_champion()

    elif args.command == "status":

        result = registry_status()

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

