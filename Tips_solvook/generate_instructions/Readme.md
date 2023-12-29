**zip파일 간단 설명**

1. python mk_instructions.py --data_path './100_Solvook_handout_DB_english.xlsx' --example
 - example 인자 지정시 example(number).json 파일들은 instruction(number) 을 개별로 저장

2. data-path에 지정된 파일경로는 google docs 공유된 excel 파일사용(11.20 기준)
 - 새롭게 다운받을 경우 excel의 5번째 sheet인 handout_type에서 skill, method 종류를 가져올 때 method 확인 요망
 (띄어쓰기나 참조오류 발생한 경우 존재)  

3. generate_instruction_textbook.ipynb는 데이터셋 전처리, 생성과정을 cell단위로 확인가능
 - handout db에서 질문 칼럼에 숫자나 기호제거, 긴 undersocre 를 짧은 것으로 대체, skill과 method 표시형식 변경
 - handout db에서 유사질문 찾기, paragraph db와 paragraph list를 merge

4. 최종 생성 파일은 Solvook_instructions.json파일에 저장되었고 prompt style은 instruction example.txt(GI LM 학습에 사용된 instruction 몇 가지를 추출한 txt파일)를 참조하여 만듬