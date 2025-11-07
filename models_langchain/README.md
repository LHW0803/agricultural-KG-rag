# Knowledge Graph RAG vs Basic LLM Comparison

농업 지식 그래프 기반 RAG와 일반 LLM의 성능을 비교하는 실험 프레임워크입니다.

## 프로젝트 구조

```
models_langchain/
├── base/
│   ├── base_model.py      # 추상 기본 클래스
│   └── evaluator.py       # 평가 메트릭 시스템
├── models/
│   ├── basic_llm.py       # 일반 LLM 모델 (구현 예정)
│   └── graph_rag.py       # Knowledge Graph RAG 모델 (구현 예정)
├── utils/
│   ├── neo4j_connector.py # Neo4j 연결 관리
│   └── data_loader.py     # QA 데이터셋 로더
├── config/
│   └── settings.py        # 설정 관리
├── results/               # 실험 결과 저장
├── main.py               # 메인 실행 파일 (구현 예정)
└── requirements.txt      # 필요 패키지
```

## 설치 및 설정

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
cp .env.example .env
```

`.env` 파일에서 다음 값들을 설정하세요:
- `OPENAI_API_KEY`: OpenAI API 키
- Neo4j 연결 정보 (필요시)

### 3. 데이터 준비
- QA 데이터셋: `qa_dataset.json` (부모 디렉터리에 위치)
- Neo4j Knowledge Graph: Agriculture_KnowledgeGraph 프로젝트의 데이터

## 주요 기능

### BaseModel 추상 클래스
- 모든 모델이 상속하는 기본 인터페이스
- 통일된 평가 및 통계 수집

### Evaluator
- BLEU, ROUGE, Cosine Similarity 메트릭
- 모델 간 성능 비교
- 결과 시각화 및 저장

### Neo4jConnector
- Neo4j 데이터베이스 연결
- 엔티티 검색 및 관계 탐색
- 컨텍스트 정보 추출

### DataLoader
- QA 데이터셋 로드 및 관리
- 데이터 분할 및 샘플링
- 통계 정보 제공

## 사용법

### 기본 사용 예제

```python
from utils import DataLoader, Neo4jConnector
from config import Settings
from base import Evaluator

# 설정 로드
settings = Settings.from_env()

# 데이터 로드
data_loader = DataLoader(settings.QA_DATASET_PATH)

# Neo4j 연결
neo4j = Neo4jConnector(settings.NEO4J_URI, 
                       settings.NEO4J_USERNAME, 
                       settings.NEO4J_PASSWORD)

# 평가기 초기화
evaluator = Evaluator()
```

## 개발 상태

- [x] 프로젝트 구조 설계
- [x] 기본 클래스 구현
- [x] 데이터 로더 구현
- [x] Neo4j 연결 모듈
- [x] 평가 시스템
- [ ] BasicLLM 모델 구현
- [ ] GraphRAG 모델 구현
- [ ] 메인 실험 스크립트
- [ ] 결과 시각화

## 라이선스

이 프로젝트는 연구용으로 제작되었습니다.