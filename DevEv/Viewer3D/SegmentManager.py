import os
import re
import numpy as np
import pkg_resources


class SegmentManager():
    def __init__(self):
        self.timestamps = self.read_timestamps()
        self.current = None
        self.category_mapping = {"c":"mat self play", "p":"parent play", "r":"room self play"}
        self.data_current = None
        
    def setCurrent(self, segment, name = ""):
        self.current = segment
        self.data_current = segment
        sess_name = re.findall(r'\d\d_\d\d', name)
        if len(sess_name) == 0 or not sess_name[0] in self.timestamps: 
            self.current = segment
            return
        info = self.timestamps[sess_name[0]]
        temp, start = [], []
        for k, list_seg in info.items():
            for (s,e) in list_seg:
                start.append(s)
                temp.append((self.category_mapping[k],s,e)) 
        self.current, index = [], np.argsort(start)
        for i in index:
             self.current.append(temp[i])
        return 
        
    def read_timestamps(self):
        
        # 'c':mat self play, 'p':parent play, 'r':room self play
        filename = pkg_resources.resource_filename('DevEv', 'metadata/DevEvData_2024-02-02.csv')
        with open(filename) as f:
            text = f.readlines()
        
        text = [l.split(",") for l in text[1:]]
        record = {}
        for data in text:
            if data[1] not in record:
                # Processed flag: False means the the method has not been processed yet
                record[data[1]] = {}
            if len(data) <= 25: category = data[-3]
            else: category = data[-6]
            if category in ['c', 'r', 'p']:
                if len(data) <= 25:
                    onset, offset = int(data[-2]), int(data[-1])
                else:
                    onset, offset = int(data[-5]), int(data[-4])
                if category not in record[data[1]]: record[data[1]][category] = []
                onset, offset = 29.97*onset/1000.0, 29.97*offset/1000.0
                record[data[1]][category].append((int(onset), int(offset)))
        return record