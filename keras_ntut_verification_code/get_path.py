import pandas as pd
import os


dirs = 'output'
dic = {}
targets = os.listdir(dirs)
targets = list(map(lambda x : dirs+'/'+ x,targets))
for path in targets:
    files = os.listdir(path)
    files = list(map(lambda x : path+'/'+x,files))
    for file_name in files:
        dic[file_name] = path.replace("output/","")
data = pd.DataFrame.from_dict(dic,orient='index')
data = data.reset_index()
data.columns = ['path','target']
# print(data)
data.to_csv("path.csv")
