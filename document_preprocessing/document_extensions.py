import os
import argparse
from collections import Counter

# 문서로 취급할 확장자 목록 (필요하면 추가 가능)
DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', 
                       '.xls', '.xlsx', '.txt', '.md', '.hwp', '.csv',
                       '.jpg',
                       'none'}

def scan_document_extensions(folder_path):
    extension_counter = Counter()
    non_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext in DOCUMENT_EXTENSIONS:
                extension_counter[ext] += 1
            else :
                extension_counter['none'] += 1
                non_list.append(file)

    return extension_counter, non_list

def main():
    parser = argparse.ArgumentParser(description="폴더 내 문서 확장자 종류 및 개수 확인")
    parser.add_argument("--dir", required=True, help="탐색할 폴더 경로를 입력하세요.")
    args = parser.parse_args()

    folder_path = args.dir

    if not os.path.isdir(folder_path):
        print("❌ 유효한 폴더 경로가 아닙니다.")
        return

    result, non_list = scan_document_extensions(folder_path)

    if result:
        print("✅ 문서 확장자별 개수:")
        for ext, count in result.items():
            print(f"{ext}: {count}개")
        print(f"총 문서 개수: {sum(result.values())}개")
        print(f"문서 확장자가 없는 파일 목록: {non_list}")
    else:
        print("📂 문서 파일이 발견되지 않았습니다.")

if __name__ == "__main__":
    main()