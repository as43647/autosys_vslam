from hybrid_slam import build_parser, build_slam


def main():
    args = build_parser().parse_args()
    build_slam(args).process()


if __name__ == "__main__":
    main()
