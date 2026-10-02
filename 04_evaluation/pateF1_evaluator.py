import numpy as np
import pandas as pd

class PateF1Evaluator:
    # Initialization of the Evaluator directly with the synopsis DataFrame
    def __init__(self, synopsis_df, buffer_zone=12, total_measurements=25000):
        self.synopsis_df = synopsis_df
        
        # Extraction of unique alarm timestamps (End_Index)
        raw_alarms = self.synopsis_df['End_Index'].unique().tolist()
        
        # The final forced flush at the end of the data stream is not an alarm.
        last_index = total_measurements - 1
        if last_index in raw_alarms:
            raw_alarms.remove(last_index)
            
        # The dynamic list of alarm time indices
        self.alarms = sorted(raw_alarms)
        
        # The static list of time indices representing the actual epoch changes
        self.drifts = [
            500, 1000, 2000, 3000, 3400, 3700, 4000, 6000, 6800, 7500, 
            8000, 11000, 12000, 13000, 14000, 14800, 15400, 15800, 
            16000, 18000, 19500, 21000, 22500, 24000
        ]
        
        # The buffer zone
        self.buffer_zone = buffer_zone

    # Execution of the PATE-F1 weight calculation and absolute alarm counting
    def _calculate_weights(self):
        w_TP = 0.0
        w_FP = 0.0
        w_FN = 0.0
        
        # Counters for the absolute operational metrics
        count_TP = 0 
        
        # Loop through each actual concept drift to evaluate detection performance
        for t_k in self.drifts:
            # Definition of the upper bound of the buffer zone for the current Epoch transition
            upper_bound = t_k + self.buffer_zone
            
            # Isolation of the alarms that fall within the current buffer zone
            valid_alarms = [t_l for t_l in self.alarms if t_k <= t_l <= upper_bound]
            
            if valid_alarms:
                # Selection of the first chronologically recorded alarm within the zone
                t_l = valid_alarms[0]
                
                # Calculation of the true positive weight (wTP) based on detection speed
                current_w_tp = (self.buffer_zone - (t_l - t_k)) / self.buffer_zone
                w_TP += current_w_tp
                
                # Calculation of the complementary false positive weight (wFP)
                w_FP += (1.0 - current_w_tp)
                
                # Increment the absolute True Alarms counter
                count_TP += 1
                
                # Removal of the selected alarm from the list to prevent re-evaluation.
                self.alarms.remove(t_l)
            else:
                # In case of a missed anomaly, the false negative weight is penalized
                w_FN += 1.0
                
        # The length of the remaining list exactly represents the absolute False Alarms
        count_FP = len(self.alarms)
        
        # Each remaining alarm contributes a maximum weight of 1.0 to wFP.
        w_FP += float(count_FP)
        
        return w_TP, w_FP, w_FN, count_TP, count_FP

    # Calculation of the final performance metrics (Precision, Recall, F1-Score)
    def _calculate_metrics(self, w_TP, w_FP, w_FN):
        # Precision calculation
        if (w_TP + w_FP) > 0:
            precision = w_TP / (w_TP + w_FP)
        else:
            precision = 0.0
            
        # Recall calculation
        if (w_TP + w_FN) > 0:
            recall = w_TP / (w_TP + w_FN)
        else:
            recall = 0.0
            
        # F1-Score calculation (Harmonic mean of Precision and Recall)
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        return precision, recall, f1_score

    # Main orchestration method to execute the evaluation process
    def run(self):
        # Execution of the internal weight calculation and counting method
        w_TP, w_FP, w_FN, count_TP, count_FP = self._calculate_weights()
        
        # Execution of the metrics calculation method
        precision, recall, f1_score = self._calculate_metrics(w_TP, w_FP, w_FN)
        
        # Return a dictionary containing the absolute counts and the final evaluation metrics
        return {
            'True_Alarms': count_TP,
            'False_Alarms': count_FP,
            'w_TP': round(w_TP, 2),
            'w_FP': round(w_FP, 2),
            'w_FN': round(w_FN, 2),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1_Score': round(f1_score, 4)
        }