from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time

class BaseModel(ABC):
    """Base class for all QA models"""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.response_times = []
        
    @abstractmethod
    def initialize(self):
        """Initialize the model and necessary connections"""
        pass
    
    @abstractmethod
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question and return response with metadata
        
        Args:
            question: The input question
            
        Returns:
            Dict containing:
                - answer: Generated answer
                - context: Retrieved context (if applicable)
                - response_time: Time taken to generate answer
                - metadata: Additional information
        """
        pass
    
    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple questions"""
        results = []
        for question in questions:
            start_time = time.time()
            result = self.answer_question(question)
            result['response_time'] = time.time() - start_time
            self.response_times.append(result['response_time'])
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, float]:
        """Get model performance statistics"""
        if not self.response_times:
            return {}
        
        return {
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'total_queries': len(self.response_times)
        }