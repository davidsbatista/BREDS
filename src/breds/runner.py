from argparse import ArgumentParser, RawDescriptionHelpFormatter

from breds.bootstrapping import BREDS
from breds.bootstrapping_parallel import BREDSParallel


def create_args() -> ArgumentParser:  # pylint: disable=missing-function-docstring
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parameters", help="file with bootstrapping configuration parameters", type=str, required=True
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
        required=True,
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
    parser.add_argument(
        "--parallel",
        default=False,
        action="store_true",
        help="whether to run the clustering in parallel or not",
    )
    parser.add_argument("--num_cores", type=int, default=0, help="number of cores to use for parallel processing")

    return parser


def main() -> None:  # pylint: disable=missing-function-docstring
    parser = create_args()
    args = parser.parse_args()

    if args.parallel:
        print("Running in parallel")
        breads = BREDSParallel(
            args.parameters, args.positive_seeds, args.negative_seeds, args.similarity, args.confidence
        )
    else:
        breads = BREDS(args.parameters, args.positive_seeds, args.negative_seeds, args.similarity, args.confidence)

    if args.sentences.endswith(".pkl"):
        print("Loading pre-processed sentences", args.sentences)
        breads.init_bootstrap(processed_tuples=args.sentences)
    else:
        breads.generate_tuples(args.sentences)
        breads.init_bootstrap(processed_tuples=None)


if __name__ == "__main__":
    main()
