import sys

from breds.breds import BREDS


def main() -> None:  # pylint: disable=missing-function-docstring
    if len(sys.argv) != 7:
        print("\nBREDS.py parameters sentences positive_seeds negative_seeds similarity confidence\n")
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = float(sys.argv[5])
        confidence = float(sys.argv[6])

        breads = BREDS(configuration, seeds_file, negative_seeds, similarity, confidence)

        if sentences_file.endswith(".pkl"):
            print("Loading pre-processed sentences", sentences_file)
            breads.init_bootstrap(processed_tuples=sentences_file)
        else:
            breads.generate_tuples(sentences_file)
            breads.init_bootstrap(processed_tuples=None)


if __name__ == "__main__":
    main()
