# Agricultural Knowledge Graph RAG Research Project

농업 지식 그래프 기반 RAG와 일반 LLM의 성능을 비교하는 연구 프로젝트입니다.

## 프로젝트 구조

```
model/
├── Agriculture_KnowledgeGraph/    # 농업 지식 그래프 (기존 프로젝트)
│   ├── demo/                      # Django 웹 애플리케이션
│   ├── KNN_predict/              # KNN 기반 엔티티 분류
│   ├── wikidataSpider/           # 관계 추출 및 데이터 처리
│   └── ...
├── models_langchain/             # LLM 비교 실험 프레임워크
│   ├── base/                     # 기본 클래스 및 평가기
│   ├── models/                   # 모델 구현체
│   ├── utils/                    # 유틸리티 (Neo4j 연결 등)
│   └── config/                   # 설정 관리
├── qa_dataset.json              # QA 데이터셋
├── load_qa_dataset.py           # Hugging Face 데이터셋 로더
└── CLAUDE.md                    # Claude Code 가이드
```

## 주요 구성 요소

### 1. Agriculture_KnowledgeGraph
- **출처**: https://github.com/qq547276542/Agriculture_KnowledgeGraph
- **기능**: 농업 엔티티 및 관계를 포함하는 Neo4j 지식 그래프
- **데이터**: 
  - 113,000+ 농업 엔티티 (HudongItem)
  - 96,000+ 추가 노드 (NewNode)
  - 관계, 속성, 기후 데이터

### 2. models_langchain
- **목적**: Graph RAG vs Basic LLM 성능 비교
- **특징**: 
  - 객체 지향적 모델 구조
  - 통합된 평가 시스템
  - Neo4j 연동
  - 다양한 평가 메트릭 (BLEU, ROUGE, Cosine Similarity)

### 3. QA Dataset
- **출처**: Hugging Face (hjoon721/final_output2)
- **형식**: Question-Answer 쌍
- **용도**: 모델 성능 평가 기준

## 설치 및 설정

### 1. 환경 요구사항
- Python 3.8+
- Neo4j 5.0+
- OpenAI API 키

### 2. 패키지 설치
```bash
cd models_langchain
pip install -r requirements.txt
```

### 3. 환경 설정
```bash
cd models_langchain
cp .env.example .env
# .env 파일에 OpenAI API 키 입력
```

### 4. Neo4j 데이터 로드
Agriculture_KnowledgeGraph의 README.md 참조하여 CSV 데이터를 Neo4j에 임포트

## 사용법

### QA 데이터셋 준비
```bash
python load_qa_dataset.py
```

### 모델 비교 실험 (개발 중)
```bash
cd models_langchain
python main.py
```

## 연구 목표

1. **기본 LLM**: 컨텍스트 없이 질문 답변
2. **Graph RAG**: Neo4j 지식 그래프 기반 컨텍스트 활용 답변
3. **성능 비교**: 정확도, 응답 품질, 처리 시간 등 종합 분석

## 개발 상태

- [x] 프로젝트 구조 설계
- [x] QA 데이터셋 로드
- [x] Neo4j 연결 모듈
- [x] 평가 시스템 기반 구축
- [ ] BasicLLM 모델 구현
- [ ] GraphRAG 모델 구현
- [ ] 실험 실행 및 결과 분석
- [ ] 논문 작성

## 참고 자료

- [Agriculture Knowledge Graph](https://github.com/qq547276542/Agriculture_KnowledgeGraph)
- [LangChain Documentation](https://docs.langchain.com/)
- [Neo4j Graph Database](https://neo4j.com/docs/)

## 라이선스

이 프로젝트는 연구 목적으로 제작되었습니다.