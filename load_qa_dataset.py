from datasets import load_dataset
import json

# Hugging Face 데이터셋 로드
dataset = load_dataset("hjoon721/final_output2")

# 데이터셋 구조 확인
print("Dataset structure:", dataset)
print("\nTrain dataset columns:", dataset['train'].column_names)
print("\nFirst example:")
first_example = dataset['train'][0]
print(json.dumps(first_example, indent=2, ensure_ascii=False))

# QA 형식으로 변환
def convert_to_qa_format(dataset):
    qa_data = []
    for item in dataset['train']:
        qa_pair = {
            'question': item['input'],
            'ground_truth': item['output'],
            'id': item.get('id', None)  # ID가 있으면 포함
        }
        qa_data.append(qa_pair)
    return qa_data

# 변환 실행
qa_dataset = convert_to_qa_format(dataset)

# 결과 확인
print(f"\n총 QA 쌍 개수: {len(qa_dataset)}")
print("\n첫 5개 QA 예시:")
for i, qa in enumerate(qa_dataset[:5], 1):
    print(f"\n{i}. Question: {qa['question'][:100]}...")
    print(f"   Ground Truth: {qa['ground_truth'][:100]}...")

# JSON 파일로 저장
with open('qa_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

print("\nQA 데이터셋이 'qa_dataset.json' 파일로 저장되었습니다.")