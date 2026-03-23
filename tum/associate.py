import argparse
import sys
import os
import numpy

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    # 使用 list comprehension 處理資料，過濾掉註解與空行
    raw_list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    formatted_list = [(float(l[0]),l[1:]) for l in raw_list if len(l)>1]
    return dict(formatted_list)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    """
    # 【關鍵修正】在 Python 3 中，必須將 keys() 轉換為 list 才能使用 .remove()
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    
    # 按照時間差從小到大排序，確保優先匹配最接近的幀
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('first_file', help='first text file (format: timestamp data)')
    parser.add_argument('second_file', help='second text file (format: timestamp data)')
    parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)

    matches = associate(first_list, second_list, float(args.offset), float(args.max_difference))    

    if args.first_only:
        for a, b in matches:
            print("%f %s"%(a, " ".join(first_list[a])))
    else:
        for a, b in matches:
            # 修正輸出的格式化字符串
            print("%f %s %f %s"%(a, " ".join(first_list[a]), b - float(args.offset), " ".join(second_list[b])))

            # python associate.py D:\dataset\tum\rgbd_dataset_freiburg3_walking_xyz\rgb.txt D:\dataset\tum\rgbd_dataset_freiburg3_walking_xyz\depth.txt > associations.txt