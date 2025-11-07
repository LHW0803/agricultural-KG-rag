from typing import List, Dict, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class Evaluator:
    """Evaluator for comparing model outputs with ground truth"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tfidf_vectorizer = TfidfVectorizer()
        
    def evaluate_single(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate a single prediction against ground truth"""
        metrics = {}
        
        # BLEU Score
        reference = [ground_truth.split()]
        candidate = prediction.split()
        metrics['bleu'] = sentence_bleu(reference, candidate)
        
        # ROUGE Scores
        rouge_scores = self.rouge_scorer.score(ground_truth, prediction)
        metrics['rouge1_f'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2_f'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL_f'] = rouge_scores['rougeL'].fmeasure
        
        # Cosine Similarity
        try:
            vectors = self.tfidf_vectorizer.fit_transform([ground_truth, prediction])
            metrics['cosine_similarity'] = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            metrics['cosine_similarity'] = 0.0
            
        # Exact Match
        metrics['exact_match'] = 1.0 if prediction.strip() == ground_truth.strip() else 0.0
        
        return metrics
    
    def evaluate_batch(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """Evaluate a batch of predictions"""
        all_metrics = []
        
        for pred, truth in zip(predictions, ground_truths):
            metrics = self.evaluate_single(pred, truth)
            all_metrics.append(metrics)
        
        # Calculate averages
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'std_{key}'] = np.std([m[key] for m in all_metrics])
        
        return {
            'individual_scores': all_metrics,
            'aggregate_scores': avg_metrics,
            'total_samples': len(predictions)
        }
    
    def compare_models(self, results_dict: Dict[str, List[Dict[str, Any]]], ground_truths: List[str]) -> Dict[str, Any]:
        """Compare results from multiple models"""
        comparison = {}
        
        for model_name, results in results_dict.items():
            predictions = [r['answer'] for r in results]
            evaluation = self.evaluate_batch(predictions, ground_truths)
            
            # Add timing information
            response_times = [r.get('response_time', 0) for r in results]
            evaluation['avg_response_time'] = np.mean(response_times) if response_times else 0
            
            comparison[model_name] = evaluation
            
        return comparison
    
    def save_results(self, comparison: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
            
    def print_summary(self, comparison: Dict[str, Any]):
        """Print a summary of the comparison"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        for model_name, results in comparison.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            aggregate = results['aggregate_scores']
            print(f"  BLEU Score: {aggregate['avg_bleu']:.4f} (±{aggregate['std_bleu']:.4f})")
            print(f"  ROUGE-1 F1: {aggregate['avg_rouge1_f']:.4f} (±{aggregate['std_rouge1_f']:.4f})")
            print(f"  ROUGE-L F1: {aggregate['avg_rougeL_f']:.4f} (±{aggregate['std_rougeL_f']:.4f})")
            print(f"  Cosine Sim: {aggregate['avg_cosine_similarity']:.4f} (±{aggregate['std_cosine_similarity']:.4f})")
            print(f"  Exact Match: {aggregate['avg_exact_match']:.4f}")
            print(f"  Avg Response Time: {results.get('avg_response_time', 0):.3f}s")
            
        print("\n" + "="*80)