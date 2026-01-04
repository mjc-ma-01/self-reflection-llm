### FRAMEWORK:

## 数据生成
    -generate.py: 为最初版本的单条数据生成测试代码。集合了多种攻击方法的demo。
    -generate_reflection_benign_data.py: 通过一个 INPUT_PATH 同时读取item和benign_item作为初始数据，并分别用两次prompt LLM生成reflection、exploration和continue续写数据，最后组合成一个完整的反思回答答案。
    --generate_reflection_data.py:
    
    -DRA
        -src/attack
            -test_openai.py:
            -clean_data.py:
## 训练模型    
    -sft.bash: 训练脚本，包含训练参数
    -sft_llm_cot.py: 训练核心代码，包含数据集加载和模型训练存储

## 评测
    -eval_llm.py: 评估模型性能，传入数据集名称，将模型回答存入results/model_answer文件夹下的对应文件中。
    -eval_json.py:评估模型回答的有害性。传入模型答案路径，按照{'question':xxx}的格式组织进行多种方式的答案有害性评估，harmbench评估本质是prompt LLM来进行有害性二元分类。
    
## 结果文件
    -results文件夹中包含
        -DRA_processed: DRA攻击，攻击llama3.1-8b-instruct生成的攻击成功数据
        -model_answer: 训练好的反思模型回答各个数据集的答案文件，
        （比如"strongreject.json"为模型回答Strongreject数据集原始答案，"strongreject_check.json"为经过harmbench、三 种字符串匹配检查之后的结果-True为包含危险回答，False为答案无害）


### TODO：
涌现什么泛化性能？
模型内部参数发生什么变化？
