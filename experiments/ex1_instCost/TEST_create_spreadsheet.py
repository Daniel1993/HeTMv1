#!/usr/bin/env python
import os
import sys
import glob
import csv
import numpy as np
from xlsxwriter.workbook import Workbook

class DataSpreadSheet:
  def __init__(self, data_dir = ".", nb_samples = 5, workbook_name = "inst_data.xlsx"):
    self.data_dir = data_dir
    self.nb_samples = nb_samples
    # self.HTM_4MB_1THR_INST      = "HTM_4MB_1THR_INST.csv"
    # self.HTM_4MB_1THR_NO_INST   = "HTM_4MB_1THR_NO_INST.csv"
    # self.HTM_600MB_1THR_INST    = "HTM_600MB_1THR_INST.csv"
    # self.HTM_600MB_1THR_NO_INST = "HTM_600MB_1THR_NO_INST.csv"
    # self.data_files = [
    #   self.HTM_4MB_1THR_INST,
    #   self.HTM_4MB_1THR_NO_INST,
    #   self.HTM_600MB_1THR_INST,
    #   self.HTM_600MB_1THR_NO_INST
    # ]
    self.data_files = sys.argv[1:]
    self.workbook = Workbook(workbook_name)
    # self.worksheet_cmp = self.workbook.add_worksheet("cmp")
    self.worksheet_data = self.workbook.add_worksheet("data")
    self.readCsv()

  def readCsv(self):
    row_df = 1
    for fname in self.data_files:
      with open(self.data_dir + "/" + fname, 'rt', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=";")
        row_add = row_df
        start_row = row_df+1
        end_row = start_row
        self.worksheet_data.write(row_df, 0, fname)
        for r, row in enumerate(reader):
          row_df += 1
          if r > 0: # ignore the sep row
            for c, col in enumerate(row):
              value = col
              if r > 1: # first row is header text
                value = float(col)
                if np.isnan(value):
                  value = 0
              self.worksheet_data.write(r+row_add, c+1, value)
            if r > 1: # to compute the averages
              end_row += 1
        #### CPU performance
        self.worksheet_data.write(end_row+1, 0, "AVERAGE")
        self.worksheet_data.write(end_row+1, 15, "=AVERAGE(P{0}:P{1})".format(start_row+2, end_row+1))
        self.worksheet_data.write(end_row+2, 0, "STD.DEV")
        self.worksheet_data.write(end_row+2, 15, "=STDEV(P{0}:P{1})".format(start_row+2, end_row+1))
        self.worksheet_data.write(end_row+3, 0, "RATIO")
        self.worksheet_data.write(end_row+3, 15, "=P{0}/P{1}".format(end_row+3, end_row+2))
        #### GPU performance
        self.worksheet_data.write(end_row+1, 0, "AVERAGE")
        self.worksheet_data.write(end_row+1, 16, "=AVERAGE(Q{0}:Q{1})".format(start_row+2, end_row+1))
        self.worksheet_data.write(end_row+2, 0, "STD.DEV")
        self.worksheet_data.write(end_row+2, 16, "=STDEV(Q{0}:Q{1})".format(start_row+2, end_row+1))
        self.worksheet_data.write(end_row+3, 0, "RATIO")
        self.worksheet_data.write(end_row+3, 16, "=Q{0}/Q{1}".format(end_row+3, end_row+2))
        #### combined performance
        self.worksheet_data.write(end_row+1, 0, "AVERAGE")
        self.worksheet_data.write(end_row+1, 18, "=AVERAGE(S{0}:S{1})".format(start_row+2, end_row+1))
        self.worksheet_data.write(end_row+2, 0, "STD.DEV")
        self.worksheet_data.write(end_row+2, 18, "=STDEV(S{0}:S{1})".format(start_row+2, end_row+1))
        self.worksheet_data.write(end_row+3, 0, "RATIO")
        self.worksheet_data.write(end_row+3, 18, "=S{0}/S{1}".format(end_row+3, end_row+2))
        row_df += 4
          

  def close(self):
    self.workbook.close()

def main():
  spreadsheet = DataSpreadSheet()
  spreadsheet.close()

if __name__ == "__main__":
  main()
