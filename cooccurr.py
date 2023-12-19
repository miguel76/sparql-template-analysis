import numpy as np
import csv

def load_cooccurrence_file(cooccurrence_filename):
    host_to_template_to_execs = {}
    with open(cooccurrence_filename) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['host'] not in host_to_template_to_execs:
                host_to_template_to_execs[row['host']] = {}
            host_to_template_to_execs[row['host']][int(row['templateId'])] = int(row['numExecs'])

    template_ids = [template_id for template_id in set([templates for host in host_to_template_to_execs for templates in host_to_template_to_execs[host]])]
    host_ids = [host for host in host_to_template_to_execs]

    host_template_execs = np.matrix([
        [
            host_to_template_to_execs[host_id][template_id]
            if template_id in host_to_template_to_execs[host_id]
            else 0
            for template_id in template_ids
        ]
        for host_id in host_ids 
    ])

    return host_template_execs.T, template_ids, host_ids