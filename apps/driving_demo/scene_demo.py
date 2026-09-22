from apps.driving_demo.scene_cli import parse_scene_args


def main() -> None:
    args = parse_scene_args()
    from apps.driving_demo.scene_runner import run_scene_demo

    run_scene_demo(args)


if __name__ == "__main__":
    main()
