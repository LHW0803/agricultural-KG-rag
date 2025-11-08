"""
Agriculture Knowledge Graph 구조 확인 (GraphRAG 모델 사용)
"""
import sys
import os
import io
import json

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models_langchain'))

from models.only_Chinese_proper_langchain_graphrag import ProperLangChainGraphRAG

print("="*80)
print("Agriculture Knowledge Graph 구조 분석")
print("="*80)

# GraphRAG 모델 초기화
model = ProperLangChainGraphRAG()

if not model.initialize():
    print("모델 초기화 실패")
    exit(1)

print("\n✓ 모델 초기화 완료\n")

# Neo4j graph 객체 가져오기
graph = model.neo4j_graph

# 1. 전체 통계
print("[1] 전체 통계")
print("-"*80)

result = graph.query("MATCH (n) RETURN count(n) as node_count")
node_count = result[0]['node_count'] if result else 0
print(f"총 노드 개수: {node_count:,}")

result = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
rel_count = result[0]['rel_count'] if result else 0
print(f"총 관계 개수: {rel_count:,}")

# 2. 노드 레이블 분포
print("\n[2] 노드 레이블 분포")
print("-"*80)

result = graph.query("""
    MATCH (n)
    RETURN labels(n) as labels, count(n) as count
    ORDER BY count DESC
    LIMIT 10
""")

for record in result:
    labels = record['labels']
    count = record['count']
    label_name = labels[0] if labels else 'No Label'
    print(f"  {label_name}: {count:,}개")

# 3. 관계 타입 분포
print("\n[3] 관계 타입 분포")
print("-"*80)

result = graph.query("""
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(r) as count
    ORDER BY count DESC
    LIMIT 10
""")

if result:
    for record in result:
        rel_type = record['rel_type']
        count = record['count']
        print(f"  {rel_type}: {count:,}개")
else:
    print("  (관계가 없거나 조회 실패)")

# 4. 노드 속성 구조
print("\n[4] HudongItem 노드 속성 구조")
print("-"*80)

result = graph.query("""
    MATCH (n:HudongItem)
    RETURN n
    LIMIT 1
""")

if result and len(result) > 0:
    node = result[0]['n']
    print(f"샘플 노드 속성:")
    for key, value in node.items():
        value_preview = str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
        print(f"  - {key}: {value_preview}")
else:
    print("  노드를 찾을 수 없습니다")

# 5. 농업 관련 노드 샘플
print("\n[5] 농업 관련 노드 샘플")
print("-"*80)

keywords = ['水稻', '农业', '种植', '肥料', '病虫害']

for keyword in keywords:
    result = graph.query(f"""
        MATCH (n:HudongItem)
        WHERE n.title CONTAINS '{keyword}'
        RETURN n.title as title, n.detail as detail
        LIMIT 2
    """)

    print(f"\n  키워드: '{keyword}' ({len(result)}개)")
    for record in result:
        title = record['title']
        detail = record.get('detail', '')
        detail_preview = detail[:80] + '...' if detail and len(detail) > 80 else detail or '(상세 정보 없음)'
        print(f"    - {title}")
        print(f"      {detail_preview}")

# 6. 노드 연결도 분석
print("\n[6] 노드 연결도 분석")
print("-"*80)

result = graph.query("""
    MATCH (n:HudongItem)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) as degree
    RETURN
        avg(degree) as avg_degree,
        max(degree) as max_degree,
        min(degree) as min_degree
""")

if result and len(result) > 0:
    stats = result[0]
    print(f"  평균 연결도: {stats['avg_degree']:.2f}")
    print(f"  최대 연결도: {stats['max_degree']}")
    print(f"  최소 연결도: {stats['min_degree']}")

# 연결도 Top 5
result = graph.query("""
    MATCH (n:HudongItem)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) as degree
    WHERE degree > 0
    RETURN n.title as title, degree
    ORDER BY degree DESC
    LIMIT 5
""")

if result:
    print("\n  연결도 Top 5:")
    for i, record in enumerate(result, 1):
        print(f"    {i}. {record['title']}: {record['degree']}개 연결")

# 7. 속성 통계
print("\n[7] HudongItem 속성 통계")
print("-"*80)

# title 속성
result = graph.query("""
    MATCH (n:HudongItem)
    WHERE n.title IS NOT NULL
    RETURN count(n) as count
""")
title_count = result[0]['count'] if result else 0
print(f"  title 속성이 있는 노드: {title_count:,}개 ({title_count/node_count*100:.1f}%)")

# detail 속성
result = graph.query("""
    MATCH (n:HudongItem)
    WHERE n.detail IS NOT NULL AND n.detail <> ''
    RETURN count(n) as count
""")
detail_count = result[0]['count'] if result else 0
print(f"  detail 속성이 있는 노드: {detail_count:,}개 ({detail_count/node_count*100:.1f}%)")

# url 속성
result = graph.query("""
    MATCH (n:HudongItem)
    WHERE n.url IS NOT NULL
    RETURN count(n) as count
""")
url_count = result[0]['count'] if result else 0
print(f"  url 속성이 있는 노드: {url_count:,}개 ({url_count/node_count*100:.1f}%)")

# 8. 관계 샘플
print("\n[8] 관계 구조 샘플 (처음 5개)")
print("-"*80)

result = graph.query("""
    MATCH (a)-[r]->(b)
    RETURN labels(a) as from_label, type(r) as rel_type, labels(b) as to_label,
           a.title as from_title, b.title as to_title
    LIMIT 5
""")

if result:
    for i, record in enumerate(result, 1):
        from_label = record['from_label'][0] if record['from_label'] else 'Unknown'
        to_label = record['to_label'][0] if record['to_label'] else 'Unknown'
        rel_type = record['rel_type']
        from_title = record['from_title'] or 'N/A'
        to_title = record['to_title'] or 'N/A'

        print(f"  {i}. ({from_label}:{from_title}) -[{rel_type}]-> ({to_label}:{to_title})")
else:
    print("  관계를 찾을 수 없습니다")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
