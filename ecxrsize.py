import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='anonymizes x-ray cases and labels output'
    )
    parser.add_argument(
        'source_folder',
        help='folder with cases'
    )
    args = parser.parse_args()

    # list of all subdirectories
    subfolders = next(os.walk(args.source_folder))[1]
    print(subfolders)

if __name__ == '__main__':
    main()
