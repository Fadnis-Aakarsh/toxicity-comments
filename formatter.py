import pandas as pd
import os
import sys

def change_format(dataframe,i):
    final = pd.DataFrame()
    alpha='a'
    
    label = 0
    output = []
    for comment in dataframe.itertuples():
        output.append([i,label,alpha,comment[1]])
        i+=1
    final = pd.DataFrame(output, columns = ['id','label','alpha','text'])
    return (final,i)

if __name__=='__main__':

    path_to_data= sys.argv[1]
    new_path_to_data = sys.argv[2]

    filelist = os.listdir(path_to_data)

    i= 1 if len(sys.argv) < 4 else int(sys.argv[3])

    for files in filelist:
        train_df = pd.read_csv(path_to_data+'/'+files, sep='\t')
        final_output,i = change_format(train_df,i)
        final_output.to_csv(new_path_to_data+'/'+files.split('.')[0]+'.tsv', sep='\t', header=True, index=False)
    