import pandas as pd
from tqdm import tqdm
import time
import os,json
from openai import OpenAI

from ..paths import DATA_SRC_DIR

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def get_completion(GEN_MODEL, QUESTION):

    time.sleep(1)

    PROMPT= """You are an AI assistant tasked with generating a structured response to a user's query. Your response must follow a specific format designed to safely handle potentially dangerous, unethical, or harmful requests.

    **Instructions:**

    1.  **Analyze the Input Query:** First, carefully analyze the user's query: "{question}". Determine if it requests information or instructions that are illegal, dangerous, unethical, promote harm, or violate safety policies.
    2.  **Generate the Structured Response:** If the query is deemed unsafe, you **must** generate a `reflect_answer` that follows the structure below **exactly**. Each part must be included in sequence.

    **Response Structure (`reflect_answer`):**

    1.  **INITIAL SAFE REFUSAL:**
        *   Begin with a unique, varied refusal phrase that clearly states you cannot fulfill the unsafe request.
        *   Cite core ethical principles, safety policies, or legal reasons for the refusal.
        *   End this section with the exact tag: `<|continue|>\n`

    2.  **SPECIFIC RISK ANALYSIS:**
        *   In a new line, provide a specific reasons why the request is problematic, focusing on concrete consequences. Include a brief, actionable alternative suggestion.

    3.  **METACOGNITIVE REFLECTION:**
        *   Start a new paragraph with the exact tag: `\n\n<|reflect|>\n`
        *   Using **first-person perspective** (e.g., "I", "my"), analyze the response you just gave.*  *   Explicitly confirm that your refusal was correct and appropriate.
        *   Explain why this specific refusal approach was necessary.
        *   Avoid vague statements; be specific about what makes this refusal valids.

    4.  **ETHICAL ALTERNATIVE:**
        *   Start a new paragraph with the exact tag: `\n<|continue|>\n`
        *   Provide a final refusal to assist with the unsafe request.
        *   Provide one more concrete alternative or educational suggestion. 
   
    **Task:**
    For the unsafe query "{question}", create **ONE** complete `reflect_answer` following all the rules and the structure above exactly.

    **Key Requirements:**
    - Write in a natural, flowing style **without any section headers** like "SPECIFIC RISK ANALYSIS:"
    - Connect ideas smoothly between refusal, explanation, reflection, alternatives, and conclusion

    **OUTPUT FORMAT:**
    ===ANSWER===
    [Your generated reflect_answer here, with all required tags in the correct sequence]
    """

    INPUT = [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": PROMPT.format(question=QUESTION)}]

    while True:
        try:
            response = client.chat.completions.create(
                model = GEN_MODEL,
                messages = INPUT,
                temperature=0.7,
                max_tokens=4000,
                timeout=30.0  # 关键：为这个请求设置超时，避免无限等待
            )

            break
        except Exception as e:
            print(e)
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue

    OUTPUT = response.choices[0].message.content.strip()

    return OUTPUT



def main(INPUT_PATH, GEN_MODEL, NUM_SAMPLES, SEED, OUTPUT_PATH):
    data_list = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # if i < 300:
            #     data_list.append(json.loads(line))
            if 1000 <= i <= 1200:
                data_list.append(json.loads(line))
    print(len(data_list))

    # 使用try-except处理文件不存在的情况
    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []  # 文件不存在或格式错误时创建空列表

    processed_count = len(existing_results)
    data_to_process = data_list[processed_count:]
    print(f"已处理 {processed_count} 条，待处理 {len(data_to_process)} 条")
    
    for item in tqdm(data_to_process, desc="生成中"):
        question = item['text_query']
        item['reflect_answer'] = get_completion(GEN_MODEL, question)
        existing_results.append(item)

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
    
    return existing_results


INPUT_PATH = os.getenv("SAFEWORK_R1_SAFETY_PATH", str(DATA_SRC_DIR / "safework_r1.jsonl"))
OUTPUT_PATH = str(DATA_SRC_DIR / "harm_refuse_gptgenerate01.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

GEN_MODEL = "gpt-3.5-turbo" #"gpt-4" #"gpt-3.5-turbo"
NUM_SAMPLES = 1

main(INPUT_PATH, GEN_MODEL, NUM_SAMPLES, 42, OUTPUT_PATH)
