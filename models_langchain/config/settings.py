import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """Configuration settings for the project"""
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 500
    
    # Neo4j Settings
    NEO4J_URI: str = "neo4j://127.0.0.1:7687"  # Updated with your Neo4j Desktop URI
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "12345678"
    
    # Data Settings
    QA_DATASET_PATH: str = r"C:\Users\leehyunwoo08\Desktop\졸논\model\qa_dataset.json"
    RESULTS_DIR: str = r"C:\Users\leehyunwoo08\Desktop\졸논\model\models_langchain\results"
    
    # Graph RAG Settings
    GRAPH_SEARCH_DEPTH: int = 2
    MAX_CONTEXT_ENTITIES: int = 10
    MAX_CONTEXT_LENGTH: int = 2000
    
    # Evaluation Settings
    BATCH_SIZE: int = 10
    TEST_SAMPLE_SIZE: Optional[int] = None  # None means use all data
    
    @classmethod
    def from_env(cls):
        """Create settings from environment variables"""
        return cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o"),
            NEO4J_URI=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            NEO4J_USERNAME=os.getenv("NEO4J_USERNAME", "neo4j"),
            NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD", "123456"),
        )
    
    def validate(self) -> bool:
        """Validate settings"""
        if not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY is not set")
            return False
        
        if not os.path.exists(self.QA_DATASET_PATH):
            print(f"Warning: QA dataset not found at {self.QA_DATASET_PATH}")
            return False
            
        return True
    
    def print_config(self):
        """Print configuration (hiding sensitive data)"""
        print("\n=== Configuration ===")
        print(f"OpenAI Model: {self.OPENAI_MODEL}")
        print(f"OpenAI API Key: {'*' * 10 if self.OPENAI_API_KEY else 'NOT SET'}")
        print(f"Neo4j URI: {self.NEO4J_URI}")
        print(f"Neo4j Username: {self.NEO4J_USERNAME}")
        print(f"QA Dataset: {self.QA_DATASET_PATH}")
        print(f"Results Directory: {self.RESULTS_DIR}")
        print("=====================\n")