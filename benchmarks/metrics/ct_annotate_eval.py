import os
import pandas as pd
import numpy as np
import argparse
import json

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from typing import Dict, List, Tuple

import logging
from typing import Dict


def setup_logger(log_file='evaluation.log'):
    """Setup logger to write to both file and console."""
    # Create logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class CellDetector:
    """Cell detection evaluation with optional matching strategies."""
    
    def __init__(self, distance_threshold: float = 20.0):
        """
        Initialize cell detector evaluator.
        
        Parameters:
        -----------
        distance_threshold : float
            Maximum distance (pixels) for valid matches
        """
        self.distance_threshold = distance_threshold
    
    def mnn_match(self, pred_points: np.ndarray, gt_points: np.ndarray) -> Tuple[List[Dict], List[int], List[int]]:
        """
        Mutual Nearest Neighbors matching.
        Matches cells only if they are each other's nearest neighbor.
        
        Parameters:
        -----------
        pred_points : np.ndarray, shape (N, 2)
            Predicted cell centroids
        gt_points : np.ndarray, shape (M, 2)
            Ground truth cell centroids
        
        Returns:
        --------
        matches : List[Dict]
            Matched pairs with keys 'pred_idx', 'gt_idx', 'distance'
        unmatched_pred : List[int]
            Unmatched prediction indices
        unmatched_gt : List[int]
            Unmatched ground truth indices
        """
        if len(pred_points) == 0 or len(gt_points) == 0:
            return [], list(range(len(pred_points))), list(range(len(gt_points)))
        
        # Build KD-trees for efficient nearest neighbor search
        tree_pred = KDTree(pred_points)
        tree_gt = KDTree(gt_points)
        
        # Find nearest neighbors in both directions
        dist_gt_to_pred, nn_gt_to_pred = tree_pred.query(gt_points)
        dist_pred_to_gt, nn_pred_to_gt = tree_gt.query(pred_points)
        
        # Find mutual nearest neighbors within threshold
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        for gt_idx, pred_idx in enumerate(nn_gt_to_pred):
            if (nn_pred_to_gt[pred_idx] == gt_idx and 
                dist_gt_to_pred[gt_idx] <= self.distance_threshold):
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': gt_idx,
                    'distance': dist_gt_to_pred[gt_idx]
                })
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        unmatched_pred = [i for i in range(len(pred_points)) if i not in matched_pred]
        unmatched_gt = [i for i in range(len(gt_points)) if i not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    
    def evaluate(self, matches: List[Dict], n_pred: int, n_gt: int) -> Dict:
        """
        Compute evaluation metrics from matching results.
        
        Parameters:
        -----------
        matches : List[Dict]
            Matched pairs from hungarian_match() or mnn_match()
        n_pred : int
            Total number of predicted cells
        n_gt : int
            Total number of ground truth cells
        
        Returns:
        --------
        metrics : Dict
            Evaluation metrics including:
            - tp, fp, fn: True/False Positive/Negative counts
            - precision, recall, f1_score: Performance metrics
            - avg_distance, std_distance: Localization accuracy
        """
        tp = len(matches)
        fp = n_pred - tp
        fn = n_gt - tp
        
        # Compute performance metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute distance statistics
        if matches:
            distances = [m['distance'] for m in matches]
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
        else:
            avg_distance = 0.0
            std_distance = 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_distance': avg_distance,
            'std_distance': std_distance,
            'n_pred': n_pred,
            'n_gt': n_gt
        }

    def evaluate_annotation(self, matches: List[Dict], 
                           pred_labels: np.ndarray, gt_labels: np.ndarray,
                           label_names: List[str] = None) -> Dict:
        """
        Evaluate cell type annotation accuracy on matched cells.
        
        Parameters:
        -----------
        matches : List[Dict]
            Matched pairs from matching methods
        pred_labels : np.ndarray
            Predicted cell type labels for all predicted cells
        gt_labels : np.ndarray
            Ground truth cell type labels for all GT cells
        label_names : List[str], optional
            List of cell type names for readable output
        
        Returns:
        --------
        annotation_metrics : Dict
            - overall_accuracy: Correct annotations / Total matches
            - per_class_precision: Precision for each cell type
            - per_class_recall: Recall for each cell type
            - per_class_f1: F1-score for each cell type
            - confusion_matrix: Confusion matrix
            - weighted_f1: F1 weighted by class frequency
            - macro_f1: Unweighted average F1 across classes
        """
        if len(matches) == 0:
            return {
                'overall_accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'n_matched': 0
            }
        
        # Extract labels for matched cells
        matched_pred_labels = np.array([pred_labels[m['pred_idx']] for m in matches])
        matched_gt_labels = np.array([gt_labels[m['gt_idx']] for m in matches])
        
         # Filter out ignored labels
        valid_gtmask = ~pd.isna(matched_gt_labels)
        valid_predmask = matched_pred_labels != "unassigned type"
        valid_mask = valid_gtmask & valid_predmask
        
        # Filter both pred and gt labels
        matched_pred_labels = matched_pred_labels[valid_mask]
        matched_gt_labels = matched_gt_labels[valid_mask]
        
        n_ignored = (~valid_mask).sum()
        # print(f"Filtered {n_ignored} matched cells with ignored labels")
        
        # Overall accuracy
        correct = sum(p == g for p, g in zip(matched_pred_labels, matched_gt_labels))
        overall_accuracy = correct / len(matched_gt_labels)
        
        # Get unique labels
        unique_labels = np.unique(np.concatenate([matched_pred_labels, matched_gt_labels]))
        
        # Compute per-class metrics
        per_class_metrics = {}
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Build confusion matrix
        for pred_label, gt_label in zip(matched_pred_labels, matched_gt_labels):
            pred_idx = label_to_idx[pred_label]
            gt_idx = label_to_idx[gt_label]
            confusion_matrix[gt_idx, pred_idx] += 1
        
        # Compute per-class precision, recall, F1
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        
        for idx, label in enumerate(unique_labels):
            tp = confusion_matrix[idx, idx]
            fp = confusion_matrix[:, idx].sum() - tp
            fn = confusion_matrix[idx, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = confusion_matrix[idx, :].sum()
            
            label_name = label_names[label] if label_names else f"Class_{label}"
            per_class_metrics[label_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            }
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            supports.append(support)
        
        # Macro F1 (unweighted average)
        macro_f1 = np.mean(f1_scores)
        
        # Weighted F1 (weighted by support)
        total_support = sum(supports)
        weighted_f1 = sum(f1 * sup for f1, sup in zip(f1_scores, supports)) / total_support
        
        return {
            'overall_accuracy': overall_accuracy,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix,
            'label_names': [label_names[l] if label_names else f"Class_{l}" 
                           for l in unique_labels],
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'n_matched': len(matches),
            'n_correct': correct
        }
    
    def compute_joint_score(self, detection_metrics: Dict, 
                           annotation_metrics: Dict,
                           detection_weight: float = 0.5) -> Dict:
        """
        Compute joint detection + annotation performance score.
        
        Parameters:
        -----------
        detection_metrics : Dict
            Output from evaluate()
        annotation_metrics : Dict
            Output from evaluate_annotation()
        detection_weight : float
            Weight for detection score (0-1), annotation weight = 1 - detection_weight
        
        Returns:
        --------
        joint_metrics : Dict
            - joint_f1: Weighted combination of detection and annotation F1
            - detection_classification_score: Detection F1 × Annotation Accuracy
            - panoptic_quality: Detection Quality × Classification Quality
        """
        det_f1 = detection_metrics['f1_score']
        ann_acc = annotation_metrics['overall_accuracy']
        ann_f1 = annotation_metrics['macro_f1']
        
        # Method 1: Weighted F1 combination
        joint_f1 = detection_weight * det_f1 + (1 - detection_weight) * ann_f1
        
        # Method 2: Detection-Classification Score (DCS)
        # Combines detection and classification directly
        dcs = det_f1 * ann_acc
        
        # Method 3: Panoptic Quality style
        # SQ (Segmentation/Detection Quality) × RQ (Recognition/Classification Quality)
        tp = detection_metrics['tp']
        fp = detection_metrics['fp']
        fn = detection_metrics['fn']
        
        # Detection quality (similar to SQ in panoptic segmentation)
        detection_quality = det_f1
        
        # Recognition quality weighted by detection success
        n_correct_class = annotation_metrics['n_correct']
        recognition_quality = n_correct_class / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        
        panoptic_quality = detection_quality * recognition_quality
        
        return {
            'joint_f1': joint_f1,
            'detection_classification_score': dcs,
            'panoptic_quality': panoptic_quality,
            'detection_f1': det_f1,
            'annotation_accuracy': ann_acc,
            'annotation_f1': ann_f1,
            'detection_weight': detection_weight
        }
    
    @staticmethod
    def report_metrics(metrics_dict: Dict, eval_method: str = 'mnn', savepath: str = None, log_file: str = None):
        
        """formatted evaluation report."""
        
        # Setup logger
        if log_file:
            logger = setup_logger(log_file)
        else:
            logger = logging.getLogger('evaluation')
            if not logger.handlers:
                logger = setup_logger('evaluation.log')
                
        logger.info("=" * 60)
        logger.info(f"Evaluation Report on {eval_method}")
        logger.info("=" * 60)
        segment = metrics_dict['segmentation']
        logger.info(f"\n{'[1] SEGMENTATION':-^80}")
        logger.info(f"\nCell Counts:")

        logger.info(f"  Predicted:      {segment['n_pred']:6d}")
        logger.info(f"  Ground Truth:   {segment['n_gt']:6d}")
        logger.info(f"\nMatching Statistics:")
        logger.info(f"  TP (matched):   {segment['tp']:6d}")
        logger.info(f"  FP (unmatched): {segment['fp']:6d}")
        logger.info(f"  FN (missed):    {segment['fn']:6d}")
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Precision: {segment['precision']:.4f}")
        logger.info(f"  Recall:    {segment['recall']:.4f}")
        logger.info(f"  F1-Score:  {segment['f1_score']:.4f}")
        logger.info("=" * 60)

        ann = metrics_dict['annotation']
        logger.info(f"\n{'[2] CELL ANNOTATION':-^80}")
        logger.info(f"  Overall Statistics:")
        logger.info(f"    Matched cells:     {ann['n_matched']:6d}")
        logger.info(f"    Correctly labeled: {ann['n_correct']:6d}")
        logger.info(f"    Accuracy:          {ann['overall_accuracy']:.4f}")
        logger.info(f"\n  Aggregate Metrics:")
        logger.info(f"    Macro F1:    {ann['macro_f1']:.4f}")
        logger.info(f"    Weighted F1: {ann['weighted_f1']:.4f}")

        joint = metrics_dict['joint']
        logger.info(f"\n{'[3] JOINT PERFORMANCE':-^80}")
        logger.info(f"  Composite Scores:")
        logger.info(f"    Joint F1:                      {joint['joint_f1']:.4f}")
        logger.info(f"    Detection-Classification:      {joint['detection_classification_score']:.4f}")
        logger.info(f"    Panoptic Quality (PQ):         {joint['panoptic_quality']:.4f} ⭐")
        logger.info(f"\n  Weighting: {joint['detection_weight']:.1f} detection / "
            f"{1 - joint['detection_weight']:.1f} annotation")

        # Summary
        logger.info(f"\n{'SUMMARY':-^80}")
        logger.info(f"  Detection F1:        {segment['f1_score']:.4f}")
        logger.info(f"  Annotation Accuracy: {ann['overall_accuracy']:.4f}")
        logger.info(f"  Overall Score (PQ):  {joint['panoptic_quality']:.4f}")
        logger.info("=" * 80 + "\n")

        if savepath:
            import pickle
            with open(savepath, 'wb') as f:
                pickle.dump(metrics_dict, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cell Type Annotation Evaluation")
    parser.add_argument('--pred_file', type=str, help='path to the predicted cell type annotation file in csv format')
    parser.add_argument('--gt_file', type=str, help='path to the ground truth cell type annotation file in csv format')
    parser.add_argument('--savepath', type=str, help='path to save the evaluation metrics')
    parser.add_argument('--logfile', type=str, help='path to save the evaluation log')
    args = parser.parse_args()

    gt_res = pd.read_csv(args.gt_file)
    gt_points = gt_res[['x', 'y']].to_numpy()
    gt_labels = gt_res['cell_type'].to_numpy()

    pred_res = pd.read_csv(args.pred_file)
    savepath = args.savepath
    logfile = args.logfile

    pred_points = pred_res[['x', 'y']].to_numpy()
    pred_labels = pred_res['cell_type'].to_numpy()
    print('Predicted cell types:', pd.Series(pred_res['cell_type'].unique()).sort_values())

    detector = CellDetector(distance_threshold=20.0)

    matches, _, _ = detector.mnn_match(pred_points, gt_points)

    segment_metrics = detector.evaluate(matches, len(pred_points), len(gt_points))

    annotation_metrics = detector.evaluate_annotation(
            matches, 
            pred_labels,
            gt_labels,
        )

    overall_metrics = detector.compute_joint_score(
            segment_metrics,
            annotation_metrics,
            detection_weight=0.5
        )

    metrics_dict = {
        'segmentation': segment_metrics,
        'annotation': annotation_metrics,
        'joint': overall_metrics
    }
    detector.report_metrics(metrics_dict, 
    args.eval_method, 
    savepath=args.savepath, 
    log_file=args.logfile)
