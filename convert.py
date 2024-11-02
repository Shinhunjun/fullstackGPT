import pypandoc

rtf_file_path = '/Users/hunjunsin/Desktop/santa.rtf'  # 변환할 .rtf 파일 경로
txt_file_path = '/Users/hunjunsin/Desktop/santa.txt'  # 저장할 .txt 파일 경로

# 변환 수행
pypandoc.convert_file(rtf_file_path, 'plain', outputfile=txt_file_path)
print("RTF 파일을 TXT 파일로 변환 완료.")