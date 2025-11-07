# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

이 프로젝트는 농업 지식 그래프(Agricultural Knowledge Graph)와 QA 데이터셋 처리를 포함하는 연구 프로젝트입니다.

## 주요 구조

### 핵심 컴포넌트
- **Agriculture_KnowledgeGraph/**: 농업 지식 그래프 메인 프로젝트
  - Neo4j 기반 지식 그래프 저장 및 쿼리
  - Django 웹 애플리케이션 (demo/)
  - KNN 기반 엔티티 분류 (KNN_predict/)
  - 웹 크롤러 (MyCrawler/, dfs_tree_crawler/)

### 데이터 처리
- **load_qa_dataset.py**: Hugging Face 데이터셋을 QA 형식으로 변환
- **qa_dataset.json**: 변환된 QA 데이터셋 저장 파일

## 개발 명령어

### 데이터셋 처리
```bash
# QA 데이터셋 로드 및 변환
python load_qa_dataset.py
```

### Django 서버 실행
```bash
cd Agriculture_KnowledgeGraph/demo
sh django_server_start.sh
# 또는
python manage.py runserver
```

### Neo4j 데이터 임포트
Neo4j에 CSV 데이터를 임포트하려면 import 폴더에 파일을 배치하고 Cypher 쿼리를 실행해야 합니다. 자세한 내용은 Agriculture_KnowledgeGraph/README.md 참조.

## 주요 기술 스택

- **백엔드**: Django, Neo4j
- **데이터 처리**: Python (pandas, datasets)
- **머신러닝**: FastText (Word2Vec), KNN 분류
- **크롤링**: Scrapy

## 데이터 파일

### 주요 데이터셋
- `hudong_pedia.csv`, `hudong_pedia2.csv`: 농업 엔티티 백과 데이터
- `labels.txt`: 5000개 이상의 수동 레이블 엔티티
- `predict_labels.txt`: KNN으로 예측한 15만개 엔티티 레이블
- `attributes.csv`: 엔티티 속성 데이터

## Neo4j 설정

Neo4j 연결 설정은 `demo/Model/neo_models.py`에서 수정 필요:
- 9번째 줄의 Neo4j 계정 정보를 사용자 환경에 맞게 변경

## 주의사항

- Neo4j와 Django 서버가 모두 실행 중이어야 웹 애플리케이션이 정상 작동
- 대용량 CSV 임포트 시 Neo4j 메모리 설정 조정 필요 (conf/neo4j.conf)
- Windows 환경에서 파일 경로는 백슬래시(\\) 사용