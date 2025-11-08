"""
Chinese-only GraphRAG 영어 질문 테스트 스크립트
qa_dataset.json의 첫 5개 영어 질문으로 테스트
(영어 질문 → 중국어 키워드 변환 → Agriculture KG 검색)
"""

import json
import time
import sys
import os
import io
from datetime import datetime

# Windows 콘솔 UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models_langchain'))

from models.only_Chinese_proper_langchain_graphrag import ProperLangChainGraphRAG

def load_qa_dataset(file_path: str, num_questions: int = 5):
    """QA 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:num_questions]

def run_test(qa_dataset, model):
    """테스트 실행"""
    results = []

    print(f"\n{'='*80}")
    print(f"Chinese-only GraphRAG 영어 질문 테스트")
    print(f"총 {len(qa_dataset)}개 질문 (영어 → 중국어 키워드 변환)")
    print(f"{'='*80}\n")

    for i, item in enumerate(qa_dataset, 1):
        question = item['question']
        ground_truth = item['ground_truth']

        print(f"\n[질문 {i}/{len(qa_dataset)}]")
        print(f"Q (English): {question}")
        print(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")

        # 시간 측정 시작
        start_time = time.time()

        try:
            # GraphRAG로 답변 생성
            result_dict = model.answer_question(question)
            answer = result_dict.get("answer", "")
            elapsed_time = time.time() - start_time

            # 메타데이터 확인
            metadata = result_dict.get("metadata", {})
            entities_found = metadata.get("entities_found", [])

            print(f"\n발견된 엔티티 ({len(entities_found)}개): {entities_found[:5]}")
            print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
            print(f"처리 시간: {elapsed_time:.2f}초")

            result = {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": answer,
                "entities_found": entities_found,
                "elapsed_time": round(elapsed_time, 2),
                "status": "success"
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            print(f"\n✗ 오류 발생: {error_msg}")
            print(f"처리 시간: {elapsed_time:.2f}초")

            result = {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": None,
                "error": error_msg,
                "elapsed_time": round(elapsed_time, 2),
                "status": "error"
            }

        results.append(result)
        print("-" * 80)

    return results

def save_results(results, output_file: str):
    """결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_with_timestamp = output_file.replace('.json', f'_{timestamp}.json')

    with open(output_file_with_timestamp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {output_file_with_timestamp}")
    return output_file_with_timestamp

def print_summary(results):
    """결과 요약 출력"""
    print(f"\n{'='*80}")
    print("테스트 결과 요약")
    print(f"{'='*80}")

    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    error = sum(1 for r in results if r['status'] == 'error')

    total_time = sum(r['elapsed_time'] for r in results)
    avg_time = total_time / total if total > 0 else 0

    # Insufficient evidence 개수
    insufficient = sum(1 for r in results if r.get('model_answer', '') == 'Insufficient evidence.')
    has_answer = success - insufficient

    print(f"총 질문 수: {total}")
    print(f"성공: {success} ({success/total*100:.1f}%)")
    print(f"  - 답변 생성: {has_answer}")
    print(f"  - Insufficient evidence: {insufficient}")
    print(f"실패: {error} ({error/total*100:.1f}%)")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"평균 처리 시간: {avg_time:.2f}초/질문")
    print(f"{'='*80}\n")

def main():
    # 설정
    QA_DATASET_PATH = "qa_dataset.json"
    OUTPUT_PATH = "test_results_english_graphrag.json"
    NUM_QUESTIONS = 5

    print("="*80)
    print("Chinese-only GraphRAG 영어 질문 테스트 스크립트")
    print("="*80)
    print("\n참고: Agriculture KG는 중국어 데이터만 포함하고 있습니다.")
    print("      영어 질문 → 중국어 키워드 변환 → KG 검색 과정을 테스트합니다.\n")

    # 1. QA 데이터셋 로드
    print(f"1. QA 데이터셋 로드 중: {QA_DATASET_PATH}")
    qa_dataset = load_qa_dataset(QA_DATASET_PATH, NUM_QUESTIONS)
    print(f"   ✓ {len(qa_dataset)}개 질문 로드 완료")

    # 2. GraphRAG 모델 초기화
    print("\n2. Chinese-only GraphRAG 초기화 중...")
    model = ProperLangChainGraphRAG()

    if not model.initialize():
        print("   ✗ 모델 초기화 실패")
        print("\n확인 사항:")
        print("   - Neo4j가 실행 중인지 확인")
        print("   - .env 파일에 OPENAI_API_KEY가 설정되어 있는지 확인")
        print("   - models_langchain/config/settings.py 설정 확인")
        return

    print("   ✓ 모델 초기화 완료")

    # 3. 테스트 실행
    print("\n3. 테스트 실행 중...")
    results = run_test(qa_dataset, model)

    # 4. 결과 저장
    print("\n4. 결과 저장 중...")
    output_file = save_results(results, OUTPUT_PATH)

    # 5. 결과 요약
    print_summary(results)

    print(f"테스트 완료! 상세 결과는 {output_file}에서 확인하세요.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
