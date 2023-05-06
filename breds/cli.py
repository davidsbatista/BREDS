from argparse import ArgumentParser, RawDescriptionHelpFormatter

from breds.bootstrapping import BREDS


def create_args() -> ArgumentParser:  # pylint: disable=missing-function-docstring
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="file with bootstrapping configuration parameters", type=str, required=False)
    parser.add_argument(
        "--word2vec",
        help="an embedding model based on word2vec, in the format of a .bin file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sentences",
        help="a text file with a sentence per line, and with at least two entities per sentence",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--positive_seeds",
        help="a text file with a seed per line, in the format, e.g.: 'Nokia;Espoo'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--negative_seeds",
        help="a text file with a seed per line, in the format, e.g.: 'Microsoft;San Francisco'",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--similarity",
        help="the minimum similarity between tuples and patterns to be considered a match",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--confidence",
        help="the minimum confidence score for a match to be considered a true positive",
        type=float,
        required=True,
    )

    return parser


def main() -> None:  # pylint: disable=missing-function-docstring
    parser = create_args()
    args = parser.parse_args()
    breads: BREDS

    breads = BREDS(args.config, args.positive_seeds, args.negative_seeds, args.similarity, args.confidence)

    if args.sentences.endswith(".pkl"):
        print("Loading pre-processed sentences", args.sentences)
        breads.init_bootstrap(args.sentences)
    else:
        breads.generate_tuples(args.sentences)
        breads.init_bootstrap()


if __name__ == "__main__":
    main()
