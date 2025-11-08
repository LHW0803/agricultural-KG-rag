"""
BasicLLM 모델 - 컨텍스트 없는 기본 GPT-4o 모델
"""
import sys
import os
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

# OpenAI 임포트
from openai import OpenAI
from dotenv import load_dotenv

# 프로젝트 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.base_model import BaseModel
from config.settings import Settings

# 환경 변수 로드
load_dotenv()

class BasicLLM(BaseModel):
    """
    BasicLLM: 외부 컨텍스트 없이 순수 GPT-4o만 사용하는 기본 모델
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("BasicLLM", config)
        self.settings = Settings.from_env()
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # 프롬프트 템플릿
        self.instruction = """As an agriculture expert, answer the farmer's questions based on their description, providing accurate, practical, and actionable advice."""
        
        self.prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{question}

### Response:
"""
        
        # 응답 로그 저장
        self.response_logs = []
        
        print(f"BasicLLM initialized with model: {self.settings.OPENAI_MODEL}")
    
    def initialize(self):
        """모델 초기화 및 연결 테스트"""
        try:
            # API 연결 테스트
            test_response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            
            print("BasicLLM: OpenAI API connection successful")
            return True
            
        except Exception as e:
            print(f"BasicLLM: API connection failed - {e}")
            return False
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (컨텍스트 없음)
        
        Args:
            question: 입력 질문
            
        Returns:
            Dict containing answer, metadata, and timing information
        """
        
        # 고유 ID 생성
        response_id = str(uuid.uuid4())
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # 프롬프트 구성
            formatted_prompt = self.prompt_template.format(
                instruction=self.instruction,
                question=question
            )
            
            # OpenAI API 호출
            api_start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS,
                top_p=0.9
            )
            
            api_end_time = time.time()
            
            # 응답 추출
            answer = response.choices[0].message.content.strip()
            
            # 메타데이터 수집
            usage_data = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            end_time = time.time()
            
            # 상세 결과 구성
            result = {
                'id': response_id,
                'timestamp': timestamp,
                'model_name': self.model_name,
                'question': question,
                'answer': answer,
                'context': None,  # BasicLLM은 컨텍스트 없음
                'response_time': end_time - start_time,
                'api_response_time': api_end_time - api_start_time,
                'metadata': {
                    'prompt_used': formatted_prompt,
                    'openai_model': self.settings.OPENAI_MODEL,
                    'temperature': self.settings.OPENAI_TEMPERATURE,
                    'max_tokens': self.settings.OPENAI_MAX_TOKENS,
                    'usage': usage_data,
                    'response_id': response.id,
                    'finish_reason': response.choices[0].finish_reason
                }
            }
            
            # 응답 로그 저장
            self.response_logs.append(result)
            
            print(f"BasicLLM: Generated response ({result['response_time']:.2f}s)")
            
            return result
            
        except Exception as e:
            error_time = time.time()
            
            # 에러 결과 구성
            error_result = {
                'id': response_id,
                'timestamp': timestamp,
                'model_name': self.model_name,
                'question': question,
                'answer': f"[ERROR] {str(e)}",
                'context': None,
                'response_time': error_time - start_time,
                'api_response_time': 0.0,
                'metadata': {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'openai_model': self.settings.OPENAI_MODEL
                }
            }
            
            self.response_logs.append(error_result)
            
            print(f"BasicLLM: Error occurred - {e}")
            
            return error_result
    
    def get_response_logs(self) -> list:
        """모든 응답 로그 반환"""
        return self.response_logs
    
    def save_logs(self, filepath: str):
        """응답 로그를 파일로 저장"""
        import json
        
        log_data = {
            'model_info': {
                'model_name': self.model_name,
                'model_type': 'BasicLLM',
                'openai_model': self.settings.OPENAI_MODEL,
                'temperature': self.settings.OPENAI_TEMPERATURE,
                'max_tokens': self.settings.OPENAI_MAX_TOKENS
            },
            'total_responses': len(self.response_logs),
            'responses': self.response_logs
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"BasicLLM: Logs saved to {filepath}")
    
    def clear_logs(self):
        """응답 로그 초기화"""
        self.response_logs = []
        print("BasicLLM: Logs cleared")


if __name__ == "__main__":
    # 테스트 코드
    model = BasicLLM()
    
    if model.initialize():
        # 테스트 질문
        test_questions = [
            "벼의 주요 병해충은 무엇인가요?",
            "토마토 재배 시 주의할 점은?",
            "유기농 농업의 장점은?"
        ]
        
        print("\n=== BasicLLM Test ===")
        
        for question in test_questions:
            print(f"\n질문: {question}")
            result = model.answer_question(question)
            print(f"답변: {result['answer'][:100]}...")
            print(f"응답시간: {result['response_time']:.2f}초")
        
        # 로그 저장
        model.save_logs("test_basic_llm_logs.json")
        
        # 통계 출력
        stats = model.get_statistics()
        print(f"\n통계: {stats}")
    else:
        print("모델 초기화 실패")