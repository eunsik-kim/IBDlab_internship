@echo off

REM combine adapter and convert model into gguf https://github.com/ggerganov/llama.cpp/discussions/2948
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
pip install peft
python combine_lora.py
python llama.cpp/convert.py ./llama2_with_mk3 --outfile ./llama2_with_mk3.gguf --outtype q8_0

REM test gguf model
pip install llama-cpp-python
python test_gguf.py