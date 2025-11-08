from typing import List, Dict, Any, Optional
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import time
from datetime import datetime
import jieba  # 중국어 토크나이저

class Evaluator:
    """Enhanced evaluator for comparing model outputs with ground truth and KG utilization"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)  # 중국어는 stemmer 사용 안 함
        self.tfidf_vectorizer = TfidfVectorizer()
        self.evaluation_logs = []

    def tokenize_chinese(self, text: str) -> List[str]:
        """중국어 텍스트를 jieba로 토큰화"""
        return list(jieba.cut(text))

    def calculate_rouge_manual(self, pred_tokens: List[str], gt_tokens: List[str]) -> Dict[str, float]:
        """수동으로 ROUGE 점수 계산 (중국어 지원)"""
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)

        # ROUGE-1: unigram overlap
        overlap = len(pred_set & gt_set)
        if len(gt_set) == 0:
            recall = 0.0
        else:
            recall = overlap / len(gt_set)

        if len(pred_set) == 0:
            precision = 0.0
        else:
            precision = overlap / len(pred_set)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def evaluate_single(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate a single prediction against ground truth (중국어 지원)"""
        metrics = {}

        # 중국어 토큰화
        gt_tokens = self.tokenize_chinese(ground_truth)
        pred_tokens = self.tokenize_chinese(prediction)

        # BLEU Score (jieba 토큰화 사용)
        reference = [gt_tokens]
        candidate = pred_tokens
        metrics['bleu'] = sentence_bleu(reference, candidate)

        # ROUGE Scores (수동 계산)
        rouge1 = self.calculate_rouge_manual(pred_tokens, gt_tokens)
        metrics['rouge1_f'] = rouge1['f1']
        metrics['rouge1_p'] = rouge1['precision']
        metrics['rouge1_r'] = rouge1['recall']

        # ROUGE-2와 ROUGE-L은 간단히 ROUGE-1으로 대체 (중국어에서는 unigram이 중요)
        metrics['rouge2_f'] = rouge1['f1']  # 간소화
        metrics['rougeL_f'] = rouge1['f1']  # 간소화

        # Cosine Similarity (토큰화된 텍스트 사용)
        gt_text = ' '.join(gt_tokens)
        pred_text = ' '.join(pred_tokens)
        try:
            vectors = self.tfidf_vectorizer.fit_transform([gt_text, pred_text])
            metrics['cosine_similarity'] = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            metrics['cosine_similarity'] = 0.0

        # Exact Match
        metrics['exact_match'] = 1.0 if prediction.strip() == ground_truth.strip() else 0.0

        return metrics
    
    def evaluate_kg_utilization(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate Knowledge Graph utilization for GraphRAG model
        
        Args:
            model_results: List of model response dictionaries with metadata
            
        Returns:
            KG utilization metrics
        """
        kg_metrics = {
            'total_queries': len(model_results),
            'kg_utilized_count': 0,
            'avg_entities_found': 0,
            'avg_kg_relations': 0,
            'avg_kg_retrieval_time': 0,
            'kg_utilization_rate': 0.0
        }
        
        if not model_results:
            return kg_metrics
        
        kg_utilized_queries = []
        entities_found_list = []
        kg_relations_list = []
        kg_retrieval_times = []
        
        for result in model_results:
            metadata = result.get('metadata', {})
            
            # Check if KG was utilized
            kg_utilized = metadata.get('kg_utilized', False)
            if kg_utilized:
                kg_metrics['kg_utilized_count'] += 1
                kg_utilized_queries.append(result)
            
            # Collect entities found
            entities_found = metadata.get('entities_found', [])
            entities_found_list.append(len(entities_found))
            
            # Collect KG relations count
            kg_relations = metadata.get('total_kg_relations', 0)
            kg_relations_list.append(kg_relations)
            
            # Collect KG retrieval time
            kg_retrieval_time = result.get('kg_retrieval_time', 0)
            kg_retrieval_times.append(kg_retrieval_time)
        
        # Calculate averages
        kg_metrics['kg_utilization_rate'] = kg_metrics['kg_utilized_count'] / kg_metrics['total_queries']
        kg_metrics['avg_entities_found'] = np.mean(entities_found_list) if entities_found_list else 0
        kg_metrics['avg_kg_relations'] = np.mean(kg_relations_list) if kg_relations_list else 0
        kg_metrics['avg_kg_retrieval_time'] = np.mean(kg_retrieval_times) if kg_retrieval_times else 0
        
        # Additional analysis for KG-utilized queries only
        if kg_utilized_queries:
            kg_utilized_entities = [len(r.get('metadata', {}).get('entities_found', [])) for r in kg_utilized_queries]
            kg_utilized_relations = [r.get('metadata', {}).get('total_kg_relations', 0) for r in kg_utilized_queries]
            
            kg_metrics['avg_entities_when_utilized'] = np.mean(kg_utilized_entities)
            kg_metrics['avg_relations_when_utilized'] = np.mean(kg_utilized_relations)
        else:
            kg_metrics['avg_entities_when_utilized'] = 0
            kg_metrics['avg_relations_when_utilized'] = 0
        
        return kg_metrics
    
    def evaluate_batch(self, predictions: List[str], ground_truths: List[str], model_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
        
        result = {
            'individual_scores': all_metrics,
            'aggregate_scores': avg_metrics,
            'total_samples': len(predictions)
        }
        
        # Add KG utilization metrics if model_results are provided
        if model_results:
            kg_metrics = self.evaluate_kg_utilization(model_results)
            result['kg_utilization_metrics'] = kg_metrics
        
        return result
    
    def compare_models(self, results_dict: Dict[str, List[Dict[str, Any]]], ground_truths: List[str]) -> Dict[str, Any]:
        """Compare results from multiple models with enhanced KG metrics"""
        comparison = {}
        
        for model_name, results in results_dict.items():
            predictions = [r['answer'] for r in results]
            
            # Enhanced evaluation with model results for KG metrics
            evaluation = self.evaluate_batch(predictions, ground_truths, model_results=results)
            
            # Add timing information
            response_times = [r.get('response_time', 0) for r in results]
            evaluation['avg_response_time'] = np.mean(response_times) if response_times else 0
            
            # Add model-specific timing details
            if any('kg_retrieval_time' in r for r in results):
                kg_retrieval_times = [r.get('kg_retrieval_time', 0) for r in results]
                api_times = [r.get('api_response_time', 0) for r in results]
                evaluation['avg_kg_retrieval_time'] = np.mean(kg_retrieval_times)
                evaluation['avg_api_response_time'] = np.mean(api_times)
            
            comparison[model_name] = evaluation
            
        return comparison
    
    def save_results(self, comparison: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
            
    def print_summary(self, comparison: Dict[str, Any]):
        """Print a comprehensive summary of the comparison including KG metrics"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        for model_name, results in comparison.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            # Quality Metrics
            aggregate = results['aggregate_scores']
            print("  QUALITY METRICS:")
            print(f"    BLEU Score: {aggregate['avg_bleu']:.4f} (±{aggregate['std_bleu']:.4f})")
            print(f"    ROUGE-1 F1: {aggregate['avg_rouge1_f']:.4f} (±{aggregate['std_rouge1_f']:.4f})")
            print(f"    ROUGE-L F1: {aggregate['avg_rougeL_f']:.4f} (±{aggregate['std_rougeL_f']:.4f})")
            print(f"    Cosine Sim: {aggregate['avg_cosine_similarity']:.4f} (±{aggregate['std_cosine_similarity']:.4f})")
            print(f"    Exact Match: {aggregate['avg_exact_match']:.4f}")
            
            # Performance Metrics
            print("\n  PERFORMANCE METRICS:")
            print(f"    Avg Response Time: {results.get('avg_response_time', 0):.3f}s")
            if 'avg_kg_retrieval_time' in results:
                print(f"    Avg KG Retrieval Time: {results['avg_kg_retrieval_time']:.3f}s")
                print(f"    Avg API Response Time: {results['avg_api_response_time']:.3f}s")
            
            # Knowledge Graph Utilization Metrics (for GraphRAG)
            if 'kg_utilization_metrics' in results:
                kg_metrics = results['kg_utilization_metrics']
                print("\n  KNOWLEDGE GRAPH UTILIZATION:")
                print(f"    KG Utilization Rate: {kg_metrics['kg_utilization_rate']:.2%}")
                print(f"    Queries with KG Context: {kg_metrics['kg_utilized_count']}/{kg_metrics['total_queries']}")
                print(f"    Avg Entities Found: {kg_metrics['avg_entities_found']:.2f}")
                print(f"    Avg KG Relations: {kg_metrics['avg_kg_relations']:.2f}")
                if kg_metrics['avg_entities_when_utilized'] > 0:
                    print(f"    Avg Entities (when KG used): {kg_metrics['avg_entities_when_utilized']:.2f}")
                    print(f"    Avg Relations (when KG used): {kg_metrics['avg_relations_when_utilized']:.2f}")
            
        print("\n" + "="*80)
    
    def create_detailed_report(self, comparison: Dict[str, Any], output_path: str):
        """
        Create a detailed evaluation report
        
        Args:
            comparison: Model comparison results
            output_path: Path to save the report
        """
        timestamp = datetime.now().isoformat()
        
        report = {
            'evaluation_timestamp': timestamp,
            'summary': {},
            'detailed_results': comparison
        }
        
        # Create summary
        for model_name, results in comparison.items():
            summary = {
                'quality_metrics': {
                    'bleu_score': results['aggregate_scores']['avg_bleu'],
                    'rouge1_f1': results['aggregate_scores']['avg_rouge1_f'],
                    'rougeL_f1': results['aggregate_scores']['avg_rougeL_f'],
                    'cosine_similarity': results['aggregate_scores']['avg_cosine_similarity'],
                    'exact_match_rate': results['aggregate_scores']['avg_exact_match']
                },
                'performance_metrics': {
                    'avg_response_time': results.get('avg_response_time', 0),
                    'total_samples': results.get('total_samples', 0)
                }
            }
            
            # Add KG metrics if available
            if 'kg_utilization_metrics' in results:
                summary['kg_utilization_metrics'] = results['kg_utilization_metrics']
            
            report['summary'][model_name] = summary
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed evaluation report saved to: {output_path}")