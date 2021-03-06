import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import PIL
import pydicom as dicom


@dataclass
class Case:
    id: str
    output_directory: str
    dicom_files: List[str]
    report_text: str
    labels: Dict[str, bool] = field(default_factory=dict)


def main():
    parser = argparse.ArgumentParser(
        description='anonymizes x-ray cases and labels output'
    )
    parser.add_argument('source_folder', help='folder with cases')
    parser.add_argument('--output', help='output', default='output')

    parser.add_argument('--p', help='process pool size', type=int, default=2)

    parser.add_argument('--csv', help='generate master csv', action='store_true')
    parser.add_argument('--merge', help='merge label csv(s)', action='store_true')

    parser.add_argument('--comprehend', help='run Comprehend Medical on reports', action='store_true')

    parser.add_argument('--entities', help='generate csv file with found entities', action='store_true')

    parser.add_argument('--tags', help='generate csv file with desired DICOM tags', action='store_true')

    parser.add_argument('--labels', help='generate labels for cases (comprehend must have already been run)', action='store_true')

    parser.add_argument('--images', help='convert dicoms to images', action='store_true')
    parser.add_argument('--ext', help='image file extension', default='jpg')
    parser.add_argument('--width', type=int, help='output width', default=500)
    parser.add_argument('--height', type=int, help='output height', default=500)

    args = parser.parse_args()

    cases = parse_source_folders(args)

    if args.comprehend:
        detect_entities_for_cases(cases, args)

    if args.entities:
        collect_entities_for_cases(cases, args)

    if args.labels:
        generate_labels_for_cases(cases, args)

    if args.images or args.tags:
        convert_dicoms_for_cases(cases, args)

    if args.csv:
        write_cases_csv(cases, args)
    
    if args.merge:
        merge_csvs(args)

def parse_source_folders(args) -> List[Case]:
    cases = []
    case_ids = os.listdir(args.source_folder)

    for case_id in case_ids:
        dicom_files = []
        case_folder = os.path.join(args.source_folder, case_id)
        if os.path.isdir(case_folder) is False:
            continue

        for file in os.listdir(case_folder):
            full_file_path = os.path.join(case_folder, file)
            if file.endswith('.txt'):
                report_text = read_file(full_file_path)
            elif file.endswith('.dcm'):
                dicom_files.append(full_file_path)

        cases.append(
            Case(
                id=case_id,
                report_text=report_text,
                dicom_files=dicom_files,
                output_directory=get_case_output_directory(case_id, args)
            )
        )

    return cases

def convert_dicoms_for_cases(cases: List[Case], args):
    if args.tags:
        csv_filename = os.path.join(args.output, 'tags.csv')
        file_already_exists = os.path.exists(csv_filename)

        HEADER = [
            "Case ID",
            "Image ID",
        ] + TAGS

        with open(csv_filename, 'a') as tags_csv_file:
            writer = csv.writer(tags_csv_file)
            if file_already_exists is False:
                writer.writerow(HEADER)

    image_start_time = datetime.now()

    with Pool(args.p) as pool:
        pool.map(partial(convert_dicoms_for_case, args=args), cases)

    print(f'Converting {len(cases)} cases took {datetime.now() - image_start_time}')

TAGS = ["ViewPosition", "PhotometricInterpretation", "SeriesDescription", "ImageComments", "AcquisitionDeviceProcessingDescription"]

def convert_dicoms_for_case(case: Case, args):
    for dicom_file in case.dicom_files:
        filename = os.path.basename(dicom_file)

        stop_before_pixels = True
        if args.images == True:
            stop_before_pixels = False

        ds = dicom.dcmread(dicom_file, stop_before_pixels=stop_before_pixels)
        if args.images:
            convert_dicom(dicom_file, output_path=case.output_directory, output_dimensions=(args.width, args.height), ext=args.ext)
        if args.tags:
            filename = os.path.basename(dicom_file)
            image_id = os.path.splitext(filename)[0]
            case_tags = extract_dicom_tags_for_case(ds)

            row = [
                case.id,
                image_id
            ] + case_tags
            csv_filename = os.path.join(args.output, 'tags.csv')
            with open(csv_filename, 'a') as tags_csv_file:
                writer = csv.writer(tags_csv_file)
                writer.writerow(row)

    print(f'Converted {case.id}')

def get_case_output_directory(id: str, args):
    case_directory = os.path.join(args.output, id)
    if not os.path.exists(case_directory):
        os.makedirs(case_directory)
    return case_directory

def convert_dicom(dicom_file: str, output_path: str, output_dimensions: Tuple[int], ext: str):
    filename = os.path.basename(dicom_file)
    ds = dicom.dcmread(dicom_file)
    image_path = filename.replace('dcm', ext)
    full_image_path = os.path.join(output_path, image_path)

    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    image_2d_scaled = np.maximum(image_2d,0) / image_2d.max()

    photometric_interpretation = ds.get("PhotometricInterpretation")
    if photometric_interpretation == "MONOCHROME1":
        image_2d_scaled = 1 - image_2d_scaled

    encode_param = []
    if ext == "jpg" or ext == "jpeg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        image_2d_scaled = image_2d_scaled * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
    elif ext == "png":
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        image_2d_scaled = image_2d_scaled * 65536.0
        image_2d_scaled = np.uint16(image_2d_scaled)
    else:
        raise("Unexpected image extension")

    resized_img = cv2.resize(image_2d_scaled, output_dimensions)
    cv2.imwrite(full_image_path, resized_img, encode_param)

def extract_dicom_tags_for_case(ds):
    dicom_tags = []
    for tag in TAGS:
        dicom_tags.append(ds.get(tag))

    return dicom_tags

def get_comprehend_medical_filename(case: Case, args):
    COMPREHEND_MEDICAL_OUTPUT_FILENAME = 'comprehendmedical.json'
    return os.path.join(case.output_directory, COMPREHEND_MEDICAL_OUTPUT_FILENAME)

def detect_entities_for_cases(cases: List[Case], args):
    count = 0

    start_time = datetime.now()

    with Pool(args.p) as pool:
        pool.map(partial(detect_entities_for_case, args=args), cases)
    
    print(f'Detecting Entities {len(cases)} cases took {datetime.now() - start_time}')

def detect_entities_for_case(case: Case, args):
    client = boto3.client(service_name='comprehendmedical')

    try:
        result = client.detect_entities(Text=case.report_text)
        output_filename = get_comprehend_medical_filename(case, args)
        file = open(output_filename, 'w')
        file.write(json.dumps(result, indent=2))
        file.close()
    except:
        print(f'Failed to detect_entities for {case.id} because {sys.exc_info()[0]}')
        pass

    print(f'Detected Entities for {case.id}')

def get_entities_csv_filename(args):
    return os.path.join(args.output, 'entities.csv')

ENTITIES_COLUMNS = ['Text', 'Category', 'Type', 'Score']
NUM_TRAITS = 3

def prepare_entities_csv(args):
    entities_csv_file = open(get_entities_csv_filename(args), 'w')

    header_columns = ['Case ID'] + ENTITIES_COLUMNS
    for i in range(NUM_TRAITS):
        header_columns.append(f'Trait {i} Name')
        header_columns.append(f'Trait {i} Score')

    entities_csv_file.write(','.join(header_columns) + '\n')
    entities_csv_file.close()

def collect_entities_for_cases(cases: List[Case], args):
    prepare_entities_csv(args)
    for case in cases:
        collect_entities_for_case(case, args)

def collect_entities_for_case(case: Case, args):
    entities = get_entities_for_case(case, args)
    text_columns = ENTITIES_COLUMNS
    lines = []
    for entity in entities:
        line = [case.id]
        for column in text_columns:
            line.append(str(entity[column]))
        
        for trait in entity['Traits']:
            line.append(trait['Name'])
            line.append(str(trait['Score']))

        lines.append(','.join(line) + '\n')
    
    entities_csv_file = open(get_entities_csv_filename(args), 'a')
    entities_csv_file.writelines(lines)
    entities_csv_file.close()

def get_entities_for_case(case: Case, args):
    comprehend_medical_filename = get_comprehend_medical_filename(case, args)
    result = json.loads(read_file(comprehend_medical_filename))
    return result['Entities']

def generate_labels_for_cases(cases: List[Case], args):
    for case in cases:
        generate_labels_for_case(case, args)
    
    write_labels_for_all_cases(cases, args.output)

def generate_labels_for_case(case: Case, args):
    entities = get_entities_for_case(case, args)

    medical_condition_entities = list(filter(is_medical_condition, entities))

    for entity in medical_condition_entities:
        diagnosis = entity['Text']
        case.labels[diagnosis] = is_positive_diagnosis(entity)
    
    write_labels_for_case(case)

def is_medical_condition(entity):
    return entity['Category'] == 'MEDICAL_CONDITION'

def is_positive_diagnosis(entity):
    traits = entity['Traits']
    for trait in traits:
        if trait['Name'] == 'NEGATION':
            return False

    return True

def write_labels_for_case(case: Case):
    write_labels_for_all_cases([case], case.output_directory)

def write_labels_for_all_cases(cases: List[Case], output_folder: str):
    possible_diagnoses_set = set()
    for case in cases:
        for label in case.labels:
            possible_diagnoses_set.add(label)
    possible_diagnoses = list(possible_diagnoses_set)
    
    labels_csv_filename = get_labels_csv_filename(output_folder)
    with open(labels_csv_filename, 'w') as labels_csv_file:
        writer = csv.writer(labels_csv_file)
        writer.writerow(['ID'] + possible_diagnoses)

        for case in cases:
            row = [case.id]
            for diagnosis in possible_diagnoses:
                exists = case.labels.get(diagnosis, False)
                row.append(exists)

            writer.writerow(row)

def get_labels_csv_filename(output_folder: str):
    return os.path.join(output_folder, 'labels.csv')

def read_file(filename: str):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def write_cases_csv(cases: Case, args):
    csv_filename = get_cases_csv_filename(args)
    file_already_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a') as labels_csv_file:
        writer = csv.writer(labels_csv_file)
        if file_already_exists is False:
            writer.writerow([
                "ID",
                "Report",
                "Source Folder",
                "Output Folder"
            ])
        for case in cases:
            writer.writerow([
                case.id,
                case.report_text,
                args.source_folder,
                case.output_directory
            ])

def get_cases_csv_filename(args):
    return os.path.join(args.output, 'cases.csv')

def get_master_csv_filename(args):
    return os.path.join(args.output, 'master.csv')

def merge_csvs(args):
    cases_df = pd.read_csv(get_cases_csv_filename(args))
    labels_df = pd.read_csv(get_labels_csv_filename(args.output))
    merged_df = pd.merge(cases_df, labels_df, how='outer', on='ID')
    merged_df.to_csv(get_master_csv_filename(args))

if __name__ == '__main__':
    main()
