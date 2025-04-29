from pathlib import Path
import tempfile
from pdf2image import convert_from_path
import os
import shutil
import argparse
from tqdm import tqdm


class ExtensionClassifier:
    def __init__(self, folder_path, output_dir):
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', 
                       '.xls', '.xlsx', '.txt', '.md', '.hwp', '.csv', '.jpg'}

    def classify_document_extensions(self):
        for root, dirs, files in os.walk(self.folder_path):
            for file in tqdm(files, desc="📋 세부 문서 분류 중..."):
                _, ext = os.path.splitext(file)
                if ext.lower() == '.pdf':
                    full_path = os.path.join(root, file)
                    
                    # 디버깅 코드는 필요하면 유지, 필요 없으면 제거
                    # import pdb; pdb.set_trace() # 1차 디버깅
                    
                    # 상대 경로 계산 - 원본 폴더 구조 유지
                    rel_path = os.path.relpath(root, self.folder_path)
                    if rel_path == ".":  # 루트 폴더인 경우
                        target_dir = self.output_dir
                    else:
                        target_dir = os.path.join(self.output_dir, rel_path)
                    
                    self.pdf_to_images(full_path, Path(target_dir))
                else : 
                    print(f"❌ 지원하지 않는 확장자입니다. : {file}")
                    pass

    def pdf_to_images(self, pdf_path, output_base_folder, dpi=300):
        # pdf_path는 문자열 또는 Path 객체일 수 있음
        pdf_path = Path(pdf_path)
        
        try:
            # PDF 파일을 이미지로 변환
            images = convert_from_path(str(pdf_path.absolute()), dpi=dpi)
            
            # 임시 디렉토리 생성
            temp_dir = Path(tempfile.mkdtemp())
            temp_pdf = temp_dir / pdf_path.name
            shutil.copy(pdf_path, temp_pdf)
            
            # output path: 파일 이름을 기반으로 출력 폴더 생성
            try:
                # 파일 이름에서 특수문자와 공백 처리
                safe_name = pdf_path.stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
                output_folder = output_base_folder / safe_name
            except Exception:
                # 이름 처리에 문제가 있으면 타임스탬프 기반 이름 사용
                from datetime import datetime
                safe_name = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_folder = output_base_folder / safe_name

            # 디버깅 코드는 필요하면 유지, 필요 없으면 제거
            # import pdb; pdb.set_trace() # 2차 디버깅
            
            # 출력 디렉토리가 없으면 생성
            try:
                output_folder.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # 권한 오류 발생 시 사용자 홈 디렉토리에 저장
                user_home = Path.home() / "test/DocLayout-YOLO/test" / safe_name
                user_home.mkdir(parents=True, exist_ok=True)
                output_folder = user_home
                print(f"⚠️ 권한 오류로 대체 경로에 저장합니다: {output_folder}")
            
            for idx, img in enumerate(images):
                try:
                    output_img_path = output_folder / f"page_{idx+1}.png"
                    img.save(output_img_path, "PNG")
                    print(f"✅ Saved: {output_img_path}")
                except Exception as e:
                    print(f"❌ 이미지 저장 중 오류: page_{idx+1}.png, 오류: {str(e)}")
                
        except Exception as e:
            print(f"❌ PDF 처리 중 오류 발생: {pdf_path.name}")
            print(f"오류 내용: {str(e)}")

    def process_pdfs(self, input_base_folder, output_base_folder):
        input_base_folder = Path(input_base_folder)
        output_base_folder = Path(output_base_folder)

        # 출력 폴더에 쓰기 권한 확인
        try:
            if not output_base_folder.exists():
                output_base_folder.mkdir(parents=True, exist_ok=True)
            # 테스트 파일 작성 시도
            test_file = output_base_folder / ".write_test"
            test_file.touch()
            test_file.unlink()  # 테스트 파일 삭제
        except PermissionError:
            # 권한 오류 시 사용자 홈 디렉토리에 대체 폴더 생성
            alt_output = Path.home() / "pdf_outputs"
            alt_output.mkdir(parents=True, exist_ok=True)
            print(f"⚠️ 출력 폴더에 쓰기 권한이 없습니다. 대체 경로를 사용합니다: {alt_output}")
            output_base_folder = alt_output

        # 원본 폴더 구조를 유지하면서 PDF 처리
        for root, dirs, files in os.walk(input_base_folder):
            for file in tqdm(files, desc="🔄 PDF 파일 처리 중..."):
                if file.lower().endswith('.pdf'):
                    pdf_path = Path(root) / file
                    
                    # 상대 경로 계산 - 원본 폴더 구조 유지
                    rel_path = os.path.relpath(root, input_base_folder)
                    if rel_path == ".":  # 루트 폴더인 경우
                        target_dir = output_base_folder
                    else:
                        target_dir = output_base_folder / rel_path
                    
                    # 대상 디렉토리 생성
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # PDF 파일을 이미지로 변환
                    self.pdf_to_images(pdf_path, target_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="폴더 내 PDF 파일을 이미지로 변환 (하위 폴더 구조 유지)")
    parser.add_argument("--input", required=True, help="원본 PDF가 들어 있는 폴더 경로")
    parser.add_argument("--output", required=True, help="이미지를 저장할 output 폴더 경로")
    args = parser.parse_args()

    extension_classifier = ExtensionClassifier(args.input, args.output)
    extension_classifier.classify_document_extensions()

    input_base_folder = Path(args.input)
    output_base_folder = Path(args.output)

    if not input_base_folder.is_dir():
        print("❌ 입력 폴더 경로가 유효하지 않습니다.")
    else:
        extension_classifier.process_pdfs(input_base_folder, output_base_folder)