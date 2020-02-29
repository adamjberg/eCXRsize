import os
import argparse
from dataclasses import dataclass
from typing import List

@dataclass
class Case:
    id: str
    dicom_files: List[str]
    report_file: str


def main():
    parser = argparse.ArgumentParser(
        description='anonymizes x-ray cases and labels output'
    )
    parser.add_argument(
        'source_folder',
        help='folder with cases'
    )
    args = parser.parse_args()
    cases = parse_source_folders(args.source_folder)
    print(cases)

def parse_source_folders(source_folder: str) -> List[Case]:
    cases = []
    case_ids = next(os.walk(source_folder))[1]

    for case_id in case_ids:
        dicom_files = []
        report_file = ""
        case_folder = os.path.join(source_folder, case_id)
        for _, _, files in os.walk(case_folder):
            for file in files:
                full_file_path = os.path.join(case_folder, file)
                if file.endswith('.txt'):
                    report_file = full_file_path
                elif file.endswith('.dicom'):
                    dicom_files.append(full_file_path)

        cases.append(Case(id=case_id, report_file=report_file, dicom_files=dicom_files))

    return cases


if __name__ == '__main__':
    main()
