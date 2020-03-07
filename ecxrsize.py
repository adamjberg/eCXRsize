import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import boto3

import cv2
import matplotlib.pyplot as plt
import PIL
import pydicom as dicom


@dataclass
class Case:
    id: str
    dicom_files: List[str]
    report_file: str

def main():
    parser = argparse.ArgumentParser(
        description='anonymizes x-ray cases and labels output'
    )
    parser.add_argument('source_folder', help='folder with cases')
    parser.add_argument('--output', help='output', default='output')

    parser.add_argument('--comprehend', help='run Comprehend Medical on reports', action='store_true')

    parser.add_argument('--images', type=bool, help='convert dicoms to images', default=False)
    parser.add_argument('--width', type=int, help='output width', default=256)
    parser.add_argument('--height', type=int, help='output height', default=256)

    args = parser.parse_args()
    cases = parse_source_folders(args.source_folder)
    for case in cases:
        if args.images:
            convert_dicoms_for_case(case, args)
        if args.comprehend:
            detect_entities_for_case(case, args)

def parse_source_folders(source_folder: str) -> List[Case]:
    cases = []
    case_ids = os.listdir(source_folder)

    for case_id in case_ids:
        dicom_files = []
        report_file = ""
        case_folder = os.path.join(source_folder, case_id)
        for file in os.listdir(case_folder):
            full_file_path = os.path.join(case_folder, file)
            if file.endswith('.txt'):
                report_file = full_file_path
            elif file.endswith('.dcm'):
                dicom_files.append(full_file_path)

        cases.append(Case(id=case_id, report_file=report_file, dicom_files=dicom_files))

    return cases

def create_output_folder_for_case(case: Case, args):
    case_directory = get_case_output_directory(case, args)
    if not os.path.exists(case_directory):
        os.makedirs(case_directory)

def convert_dicoms_for_case(case: Case, args):
    create_output_folder_for_case(case, args)
    for dicom_file in case.dicom_files:
        output_path = get_case_output_directory(case, args)
        convert_dicom(dicom_file, output_path=output_path, output_dimensions=(args.width, args.height))

def get_case_output_directory(case: Case, args):
    return os.path.join(args.output, case.id)

def convert_dicom(dicom_file: str, output_path: str, output_dimensions: Tuple[int]):
    filename = os.path.basename(dicom_file)
    ds = dicom.dcmread(dicom_file)
    pixel_array_numpy = ds.pixel_array
    image_path = filename.replace('dcm', 'jpg')
    full_image_path = os.path.join(output_path, image_path)
    resized_img = cv2.resize(pixel_array_numpy, output_dimensions)
    cv2.imwrite(full_image_path, resized_img)

COMPREHEND_MEDICAL_OUTPUT_FILENAME = 'comprehendmedical.json'

def detect_entities_for_case(case: Case, args):
    client = boto3.client(service_name='comprehendmedical')
    result = client.detect_entities(Text= 'cerealx 84 mg daily')
    output_directory = get_case_output_directory(case, args)
    filename = os.path.join(output_directory, COMPREHEND_MEDICAL_OUTPUT_FILENAME)
    file = open(filename, "w")
    file.write(json.dumps(result, indent=2))
    file.close()

if __name__ == '__main__':
    main()
