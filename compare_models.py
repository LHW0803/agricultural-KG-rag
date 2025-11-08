"""
BasicLLM vs Chinese-only GraphRAG 모델 비교 테스트
qa_dataset_chinese.json의 10개 질문으로 두 모델 평가
"""
import json
import sys
import os
import io
from datetime import datetime

# Windows 콘솔 UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models_langchain'))

from models.basic_llm import BasicLLM
from models.only_Chinese_proper_langchain_graphrag import ProperLangChainGraphRAG
from base.evaluator import Evaluator

def load_qa_dataset(file_path: str):
    """QA 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def test_basic_llm(qa_dataset):
    """BasicLLM 테스트"""
    print(f"\n{'='*80}")
    print("BasicLLM 테스트 시작")
    print(f"{'='*80}\n")

    model = BasicLLM()

    if not model.initialize():
        print("✗ BasicLLM 초기화 실패")
        return None

    print("✓ BasicLLM 초기화 완료\n")

    results = []

    for i, item in enumerate(qa_dataset, 1):
        question = item['question']
        question_id = item['id']

        print(f"[{i}/{len(qa_dataset)}] ID: {question_id}")
        print(f"Q: {question}")

        try:
            result = model.answer_question(question)
            answer = result['answer']

            print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
            print(f"시간: {result['response_time']:.2f}초\n")

            results.append(result)

        except Exception as e:
            print(f"✗ 오류: {e}\n")
            results.append({
                'id': f"error_{question_id}",
                'question': question,
                'answer': f"[ERROR] {str(e)}",
                'response_time': 0,
                'metadata': {}
            })

    return results

def test_graphrag(qa_dataset):
    """GraphRAG 테스트"""
    print(f"\n{'='*80}")
    print("Chinese-only GraphRAG 테스트 시작")
    print(f"{'='*80}\n")

    model = ProperLangChainGraphRAG()

    if not model.initialize():
        print("✗ GraphRAG 초기화 실패")
        return None

    print("✓ GraphRAG 초기화 완료\n")

    results = []

    for i, item in enumerate(qa_dataset, 1):
        question = item['question']
        question_id = item['id']

        print(f"[{i}/{len(qa_dataset)}] ID: {question_id}")
        print(f"Q: {question}")

        try:
            result_dict = model.answer_question(question)

            # None 반환 = 노드가 없어서 스킵
            if result_dict is None:
                print(f"✗ 스킵: KG에 관련 노드가 없습니다\n")
                # Evaluator를 위해 빈 답변으로 처리
                results.append({
                    'id': f"skipped_{question_id}",
                    'question': question,
                    'answer': "[NO_NODES] No relevant nodes found in Knowledge Graph",
                    'response_time': 0,
                    'metadata': {
                        'kg_utilized': False,
                        'entities_found': [],
                        'total_kg_relations': 0,
                        'node_count': 0,
                        'evidence_count': 0,
                        'has_sufficient_evidence': False
                    }
                })
            else:
                answer = result_dict.get("answer", "")
                metadata = result_dict.get("metadata", {})

                has_sufficient = metadata.get("has_sufficient_evidence", False)
                node_count = metadata.get("node_count", 0)
                evidence_count = metadata.get("evidence_count", 0)

                print(f"노드: {node_count}개 | Evidence: {evidence_count}개 | 충분한 근거: {has_sufficient}")
                print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
                print(f"시간: {result_dict.get('response_time', 0):.2f}초\n")

                results.append(result_dict)

        except Exception as e:
            print(f"✗ 오류: {e}\n")
            results.append({
                'id': f"error_{question_id}",
                'question': question,
                'answer': f"[ERROR] {str(e)}",
                'response_time': 0,
                'metadata': {}
            })

    return results

def save_individual_results(basic_results, graphrag_results, output_dir="."):
    """각 모델의 결과를 개별 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # BasicLLM 결과 저장
    basic_file = os.path.join(output_dir, f"basic_llm_results_{timestamp}.json")
    with open(basic_file, 'w', encoding='utf-8') as f:
        json.dump(basic_results, f, ensure_ascii=False, indent=2)
    print(f"BasicLLM 결과 저장: {basic_file}")

    # GraphRAG 결과 저장
    graphrag_file = os.path.join(output_dir, f"graphrag_results_{timestamp}.json")
    with open(graphrag_file, 'w', encoding='utf-8') as f:
        json.dump(graphrag_results, f, ensure_ascii=False, indent=2)
    print(f"GraphRAG 결과 저장: {graphrag_file}")

    return basic_file, graphrag_file

def main():
    # 설정
    QA_DATASET_PATH = "qa_dataset_chinese.json"

    print("="*80)
    print("BasicLLM vs Chinese-only GraphRAG 모델 비교")
    print("="*80)

    # 1. QA 데이터셋 로드
    print(f"\n1. QA 데이터셋 로드 중: {QA_DATASET_PATH}")
    qa_dataset = load_qa_dataset(QA_DATASET_PATH)
    ground_truths = [item['ground_truth'] for item in qa_dataset]
    print(f"   ✓ {len(qa_dataset)}개 질문 로드 완료")

    # 2. BasicLLM 테스트
    print("\n2. BasicLLM 테스트")
    basic_results = test_basic_llm(qa_dataset)

    if basic_results is None:
        print("BasicLLM 테스트 실패")
        return

    # 3. GraphRAG 테스트
    print("\n3. GraphRAG 테스트")
    graphrag_results = test_graphrag(qa_dataset)

    if graphrag_results is None:
        print("GraphRAG 테스트 실패")
        return

    # 4. 개별 결과 저장
    print(f"\n4. 개별 결과 저장")
    save_individual_results(basic_results, graphrag_results)

    # 5. Evaluator로 비교
    print(f"\n5. 모델 비교 평가")
    print("-" * 80)

    evaluator = Evaluator()

    # 모델 결과 구성
    models_results = {
        'BasicLLM': basic_results,
        'GraphRAG_Chinese': graphrag_results
    }

    # 비교 평가 수행
    comparison = evaluator.compare_models(models_results, ground_truths)

    # 6. 결과 출력
    evaluator.print_summary(comparison)

    # 7. 상세 리포트 저장
    print("\n6. 상세 리포트 저장")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_file = f"model_comparison_{timestamp}.json"
    evaluator.save_results(comparison, comparison_file)
    print(f"   비교 결과: {comparison_file}")

    report_file = f"evaluation_report_{timestamp}.json"
    evaluator.create_detailed_report(comparison, report_file)
    print(f"   상세 리포트: {report_file}")

    print("\n" + "="*80)
    print("평가 완료!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n평가가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
