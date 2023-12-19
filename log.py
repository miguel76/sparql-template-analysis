import numpy as np
import pandas as pd
import csv

def load_log_file(log_filename, template_clusters, template_ids, time_threshold=pd.Timedelta(1, "hour")):
    logs = pd.read_csv(log_filename, sep=',', parse_dates=['execTime'])
    logs_by_cluster = []
    for cluster_id in range(template_clusters.shape[0]):
        cluster_templates = [template_ids[i] for i in np.where(template_clusters[cluster_id])[0]]
        cluster_logs = logs[logs['templateId'].isin(cluster_templates)]
        cluster_logs_splitted = pd.concat([
            host_data.assign(
                timeDelay = host_data['execTime'].diff(),
                templateCaseId=(host_data['execTime'].diff() > time_threshold).cumsum())
            for _, host_data
            in cluster_logs.groupby('host')
        ])
        logs_by_cluster.append(pd.concat([
            host_case_data.assign(caseId = case_count)
            for case_count, (_, host_case_data)
            in enumerate(cluster_logs_splitted.groupby(['host', 'templateCaseId']))
        ]))
    return logs_by_cluster