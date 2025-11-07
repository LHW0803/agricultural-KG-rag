import json
from typing import List, Dict, Any, Optional
import random

class DataLoader:
    """Load and manage QA dataset"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to qa_dataset.json
        """
        self.data_path = data_path
        self.data = []
        self.load_data()
        
    def load_data(self):
        """Load data from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} QA pairs from {self.data_path}")
        
    def get_questions(self) -> List[str]:
        """Get all questions"""
        return [item['question'] for item in self.data]
    
    def get_ground_truths(self) -> List[str]:
        """Get all ground truth answers"""
        return [item['ground_truth'] for item in self.data]
    
    def get_qa_pairs(self) -> List[Dict[str, str]]:
        """Get all QA pairs"""
        return self.data
    
    def get_sample(self, n: Optional[int] = None, random_sample: bool = True) -> List[Dict[str, str]]:
        """
        Get a sample of QA pairs
        
        Args:
            n: Number of samples (None for all)
            random_sample: Whether to randomly sample
            
        Returns:
            List of QA pairs
        """
        if n is None or n >= len(self.data):
            return self.data
            
        if random_sample:
            return random.sample(self.data, n)
        else:
            return self.data[:n]
    
    def split_data(self, test_ratio: float = 0.2) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            test_ratio: Ratio of test data
            
        Returns:
            train_data, test_data
        """
        data_copy = self.data.copy()
        random.shuffle(data_copy)
        
        split_idx = int(len(data_copy) * (1 - test_ratio))
        train_data = data_copy[:split_idx]
        test_data = data_copy[split_idx:]
        
        return train_data, test_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        questions = self.get_questions()
        answers = self.get_ground_truths()
        
        avg_q_length = sum(len(q.split()) for q in questions) / len(questions)
        avg_a_length = sum(len(a.split()) for a in answers) / len(answers)
        
        return {
            'total_pairs': len(self.data),
            'avg_question_words': avg_q_length,
            'avg_answer_words': avg_a_length,
            'unique_questions': len(set(questions)),
            'sample_questions': questions[:3]
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]